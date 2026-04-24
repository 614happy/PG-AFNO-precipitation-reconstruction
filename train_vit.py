import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath, trunc_normal_
from tqdm import tqdm

from dataloader_1to69 import get_dataloader


def set_seed(seed: int = 614):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Config:
    data_path: str = "path/to/normalized_era5"
    output_root: str = "path/to/output/vit_baseline"
    batch_size: int = 2
    epochs: int = 30
    patience: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-5


class PatchEmbed(nn.Module):
    def __init__(self, img_size=160, patch_size=4, in_chans=1, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, num_tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)
        x = (attention @ value).transpose(1, 2).reshape(batch, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class WeatherViT(nn.Module):
    """ViT baseline used for the nonlocal comparison."""

    def __init__(
        self,
        img_size=160,
        patch_size=4,
        in_chans=1,
        out_chans=69,
        embed_dim=384,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_chans = out_chans
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [value.item() for value in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                ViTBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[idx],
                )
                for idx in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, out_chans * patch_size * patch_size)
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == 161 or x.shape[-2] == 161:
            x = x[:, :, :160, :160]
        batch, _, height, width = x.shape
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.head(x)
        h_tokens = height // self.patch_size
        w_tokens = width // self.patch_size
        x = rearrange(
            x,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=h_tokens,
            w=w_tokens,
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.out_chans,
        )
        return x


class Logger:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def write(self, message: str):
        print(message)
        with open(self.file_path, "a", encoding="utf-8") as file:
            file.write(message + "\n")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device).float()
        targets = targets.to(device).float()
        outputs = model(inputs)
        total_loss += F.mse_loss(outputs, targets).item()
    return total_loss / len(loader)


def train_vit(config: Config):
    os.makedirs(config.output_root, exist_ok=True)
    logger = Logger(os.path.join(config.output_root, "train_vit.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.write(f"Device: {device}")

    train_loader, valid_loader, test_loader = get_dataloader(
        train_batch_size=config.batch_size,
        valid_batch_size=config.batch_size,
        test_batch_size=config.batch_size,
        base_path=config.data_path,
    )

    model = WeatherViT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    best_valid = float("inf")
    best_epoch = 0
    patience_counter = 0
    checkpoint_path = os.path.join(config.output_root, "vit_best.pth")

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1:02d}", leave=False):
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)
        valid_loss = evaluate(model, valid_loader, device)
        logger.write(f"Epoch {epoch + 1:02d} | train_mse={train_loss:.6f} | valid_mse={valid_loss:.6f}")

        if valid_loss < best_valid:
            best_valid = valid_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({"model_state": model.state_dict(), "epoch": best_epoch, "valid_mse": best_valid}, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.write("Early stopping triggered.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_loss = evaluate(model, test_loader, device)

    results_path = os.path.join(config.output_root, "results_vit.txt")
    with open(results_path, "w", encoding="utf-8") as file:
        file.write("Model: vit\n")
        file.write(f"Best epoch: {best_epoch}\n")
        file.write(f"Best validation MSE: {best_valid:.6f}\n")
        file.write(f"Test MSE: {test_loss:.6f}\n")

    logger.write(f"Best epoch: {best_epoch}")
    logger.write(f"Test MSE: {test_loss:.6f}")
    logger.write(f"Saved results to: {results_path}")


if __name__ == "__main__":
    set_seed(614)
    train_vit(Config())
