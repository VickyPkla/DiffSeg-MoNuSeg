import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np

from unet import ConditionalUNet
from input_model import RRDBNetSimple
from seg_model import SegModel


# -------------------------------
# Seed & Determinism
# -------------------------------
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# -------------------------------
# Transform
# -------------------------------
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

mask_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# -------------------------------
# Dataset
# -------------------------------
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class PatchDataset(Dataset):

    def __init__(self, root_dir, patch_size=128, stride=64,
                 image_transform=None, mask_transform=None):
        self.mask_dir        = os.path.join(root_dir, "masks")
        self.image_dir       = os.path.join(root_dir, "images")
        self.patch_size      = patch_size
        self.stride          = stride          # controls overlap — stride < patch_size = overlap
        self.image_transform = image_transform
        self.mask_transform  = mask_transform

        self.filenames = sorted(os.listdir(self.mask_dir))

        # Store patch grid info for each image
        # Each entry: (filename, patch_origins, total_patches)
        # patch_origins: list of (top, left) for every patch
        self.image_info        = []
        self.cumulative_patches = []

        total = 0

        for name in self.filenames:
            image_path = os.path.join(self.image_dir, name)

            with Image.open(image_path) as img:
                W, H = img.size

            # Generate all (top, left) origins using stride
            # Only include patches that fit fully within the image
            origins = []
            for top in range(0, H - patch_size + 1, stride):
                for left in range(0, W - patch_size + 1, stride):
                    origins.append((top, left))

            # Ensure the last row and column are always covered
            # even if (H - patch_size) is not divisible by stride
            last_top  = H - patch_size
            last_left = W - patch_size

            for top in range(0, H - patch_size + 1, stride):
                if (top, last_left) not in origins:
                    origins.append((top, last_left))

            for left in range(0, W - patch_size + 1, stride):
                if (last_top, left) not in origins:
                    origins.append((last_top, left))

            if (last_top, last_left) not in origins:
                origins.append((last_top, last_left))

            total_patches = len(origins)
            self.image_info.append((name, origins, total_patches))

            total += total_patches
            self.cumulative_patches.append(total)

        self.total_patches = total
        print(f"Total patches (stride={stride}): {self.total_patches}")

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):

        # Find which image this idx belongs to
        img_idx = 0
        while idx >= self.cumulative_patches[img_idx]:
            img_idx += 1

        local_idx = idx - self.cumulative_patches[img_idx - 1] if img_idx > 0 else idx

        name, origins, _ = self.image_info[img_idx]

        mask_path  = os.path.join(self.mask_dir,  name)
        image_path = os.path.join(self.image_dir, name)

        # Load image
        image = Image.open(image_path)
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        else:
            image = image.convert("RGB")

        mask = Image.open(mask_path).convert("L")

        # Get patch origin
        top, left = origins[local_idx]
        bottom = top  + self.patch_size
        right  = left + self.patch_size

        # Crop
        image_patch = image.crop((left, top, right, bottom))
        mask_patch  = mask.crop((left, top, right, bottom))

        # Transform
        if self.image_transform:
            image_patch = self.image_transform(image_patch)

        if self.mask_transform:
            mask_patch = self.mask_transform(mask_patch)

        return mask_patch, image_patch


# -------------------------------
# Diffusion Noise
# -------------------------------
def add_noise(x, gamma_t):
    noise = torch.randn_like(x)
    noisy = torch.sqrt(gamma_t) * x + torch.sqrt(1 - gamma_t) * noise
    return noisy, noise


# -------------------------------
# Charbonnier Loss (Fixed)
# -------------------------------
def charbonnier_loss(pred, target):
    return torch.mean(torch.sqrt((pred - target) ** 2 + 1e-6))


# -------------------------------
# Find Latest Checkpoint
# -------------------------------
def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith("checkpoint_epoch_") and f.endswith(".pth"):
            try:
                epoch_num = int(f.replace("checkpoint_epoch_", "").replace(".pth", ""))
                checkpoints.append((epoch_num, os.path.join(checkpoint_dir, f)))
            except ValueError:
                continue

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


# -------------------------------
# Safe Save
# -------------------------------
def safe_save(state, path):
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


# -------------------------------
# Training
# -------------------------------
def train_ddpm(
        input_model,
        seg_model,
        unet_model,
        dataloader,
        num_timesteps=1000,
        num_epochs=500,
        lr=2e-4,
        resume_checkpoint=None):

    parameters = (
        list(unet_model.parameters()) +
        list(input_model.parameters()) +
        list(seg_model.parameters())
    )

    optimizer = torch.optim.AdamW(parameters, lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )

    beta  = torch.linspace(1e-4, 0.02, num_timesteps).to(device)
    alpha = 1 - beta
    gamma = torch.cumprod(alpha, dim=0)

    os.makedirs("checkpoints", exist_ok=True)

    start_epoch = 0
    best_loss = float("inf")

    # Resume
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming from {resume_checkpoint}")
        ck = torch.load(resume_checkpoint, map_location=device)

        unet_model.load_state_dict(ck["unet"])
        input_model.load_state_dict(ck["input_model"])
        seg_model.load_state_dict(ck["seg_model"])
        optimizer.load_state_dict(ck["optimizer"])

        start_epoch = ck["epoch"]
        best_loss = ck.get("loss", float("inf"))

        if "scheduler" in ck:
            scheduler.load_state_dict(ck["scheduler"])
        else:
            for _ in range(start_epoch):
                scheduler.step()

    input_model.train()
    seg_model.train()
    unet_model.train()

    # -------------------------------
    # Training Loop
    # -------------------------------
    for epoch in range(start_epoch, num_epochs):

        epoch_loss = 0

        for mask, image in dataloader:

            mask = mask.to(device)      # [B,1,H,W]
            image = image.to(device)    # [B,3,H,W]

            B = mask.shape[0]

            t = torch.randint(0, num_timesteps, (B,), device=device).long()
            gamma_t = gamma[t].view(B, 1, 1, 1)

            noisy_mask, noise = add_noise(mask, gamma_t)

            seg_features = seg_model(noisy_mask)
            input_features = input_model(image)

            combined = seg_features + input_features

            pred_noise = unet_model(combined, t)

            loss = charbonnier_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.6f} | LR: {lr_now:.2e}")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            path = f"checkpoints/checkpoint_epoch_{epoch+1}.pth"
            safe_save({
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "unet": unet_model.state_dict(),
                "input_model": input_model.state_dict(),
                "seg_model": seg_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()
            }, path)

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print(f"***********************   Best model saved   ***********************")
            safe_save({
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "unet": unet_model.state_dict(),
                "input_model": input_model.state_dict(),
                "seg_model": seg_model.state_dict(),
                "scheduler": scheduler.state_dict()
            }, "checkpoints/best_model.pth")


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":

    dataset = PatchDataset(
        root_dir="MonuSeg/Train",
        patch_size=64,
        stride=32,
        image_transform=image_transform,
        mask_transform=mask_transform
    )


    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    unet_model = ConditionalUNet(
        in_channels=32,
        out_channels=1
    ).to(device)

    input_model = RRDBNetSimple(
        out_channels=32
    ).to(device)

    seg_model = SegModel(
        out_channels=32
    ).to(device)

    resume_checkpoint = find_latest_checkpoint("checkpoints")

    train_ddpm(
        input_model,
        seg_model,
        unet_model,
        dataloader,
        num_timesteps=1000,
        num_epochs=300,
        lr=1e-5,
        resume_checkpoint=resume_checkpoint
    )
