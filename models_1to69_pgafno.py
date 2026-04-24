import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer=nn.GELU,
        drop: float = 0.1,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


def generate_asymmetric_threshold(
    height: int,
    width: int,
    base_threshold: float = 0.01,
    k_decay: float = 20.0,
    min_threshold: float = 5e-4,
) -> torch.Tensor:
    """Create the wavenumber-dependent threshold matrix T(k)."""
    freq_u = torch.fft.fftfreq(height, d=1.0).view(-1, 1)
    freq_v = torch.fft.rfftfreq(width, d=1.0).view(1, -1)
    k = torch.sqrt(freq_u**2 + freq_v**2) * max(height, width)
    threshold = base_threshold * torch.exp(-0.5 * (k / k_decay) ** 2)
    threshold = torch.clamp(threshold, min=min_threshold)
    return threshold.unsqueeze(0).unsqueeze(-1)


class AFNO2D(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        height: int = 20,
        width: int = 20,
        num_blocks: int = 8,
        base_threshold: float = 0.01,
        k_decay: float = 20.0,
        min_threshold: float = 5e-4,
    ) -> None:
        super().__init__()
        if hidden_size % num_blocks != 0:
            raise ValueError("hidden_size must be divisible by num_blocks.")

        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks
        self.scale = 0.02

        threshold = generate_asymmetric_threshold(
            height=height,
            width=width,
            base_threshold=base_threshold,
            k_decay=k_decay,
            min_threshold=min_threshold,
        )
        self.register_buffer("threshold_matrix", threshold)

        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size)
        )
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size)
        )
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = x
        dtype = x.dtype
        x = x.float()

        batch_size, height, width_spatial, channels = x.shape
        width_complex = width_spatial // 2 + 1

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(batch_size, height, width_complex, self.num_blocks, self.block_size)

        o1_real = F.relu(
            torch.einsum("...bi,bio->...bo", x.real, self.w1[0])
            - torch.einsum("...bi,bio->...bo", x.imag, self.w1[1])
            + self.b1[0]
        )
        o1_imag = F.relu(
            torch.einsum("...bi,bio->...bo", x.imag, self.w1[0])
            + torch.einsum("...bi,bio->...bo", x.real, self.w1[1])
            + self.b1[1]
        )
        o2_real = (
            torch.einsum("...bi,bio->...bo", o1_real, self.w2[0])
            - torch.einsum("...bi,bio->...bo", o1_imag, self.w2[1])
            + self.b2[0]
        )
        o2_imag = (
            torch.einsum("...bi,bio->...bo", o1_imag, self.w2[0])
            + torch.einsum("...bi,bio->...bo", o1_real, self.w2[1])
            + self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)

        magnitude = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-8)
        shrink_factor = F.relu(magnitude - self.threshold_matrix.unsqueeze(-2)) / magnitude
        x = x * shrink_factor.unsqueeze(-1)

        x = torch.view_as_complex(x)
        x = x.reshape(batch_size, height, width_complex, channels)
        x = torch.fft.irfft2(x, s=(height, width_spatial), dim=(1, 2), norm="ortho")
        return x.type(dtype) + bias


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        height: int = 20,
        width: int = 20,
        mlp_ratio: float = 4.0,
        drop: float = 0.1,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = AFNO2D(dim, height=height, width=width)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.filter(self.norm1(x)) + residual
        residual = x
        x = self.drop_path(self.mlp(self.norm2(x))) + residual
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int] = (160, 160),
        patch_size: tuple[int, int] = (8, 8),
        in_chans: int = 1,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class AFNONet(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int] = (160, 160),
        patch_size: tuple[int, int] = (8, 8),
        in_chans: int = 1,
        out_chans: int = 69,
        embed_dim: int = 768,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.15,
        drop_path_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.num_patches
        self.height = img_size[0] // patch_size[0]
        self.width = img_size[1] // patch_size[1]

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        drop_path_values = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    height=self.height,
                    width=self.width,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=drop_path_values[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, out_chans * patch_size[0] * patch_size[1])

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = x.reshape(batch_size, self.height, self.width, self.embed_dim)
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.norm(x)
        x = self.head(x)
        x = rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            c_out=self.out_chans,
        )
        return x


class PGAFNO(nn.Module):
    def __init__(
        self,
        in_chans: int = 1,
        out_chans: int = 69,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone or AFNONet(
            img_size=(160, 160),
            patch_size=(8, 8),
            in_chans=in_chans,
            out_chans=out_chans,
            embed_dim=768,
            depth=12,
            drop_rate=0.15,
            drop_path_rate=0.2,
        )
        self.pad = nn.ReflectionPad2d(1)
        self.refine = nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor, gt: torch.Tensor | None = None):
        if x.shape[-1] == 161 or x.shape[-2] == 161:
            x = x[:, :, :160, :160]

        x = self.backbone(x)
        x = self.pad(x)
        x = self.refine(x)

        if gt is not None:
            if gt.shape[-1] == 161 or gt.shape[-2] == 161:
                gt = gt[:, :, :160, :160]
            loss = F.mse_loss(x, gt)
            return x, loss

        return x


if __name__ == "__main__":
    model = PGAFNO(in_chans=1, out_chans=69)
    sample = torch.randn(2, 1, 160, 160)
    output = model(sample)
    num_params = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    print(f"Output shape: {output.shape}")
    print(f"Parameter count: {num_params:.2f} M")
