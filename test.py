import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from unet import ConditionalUNet
from input_model import RRDBNetSimple
from seg_model import SegModel


# -------------------------------
# Config
# -------------------------------
NUM_TIMESTEPS   = 1000
N_RUNS          = 1
THRESHOLD       = 0.7
PATCH_SIZE      = 64
STRIDE          = 32

CHECKPOINT_PATH = "checkpoints/best_model.pth"
INPUT_IMAGE     = "samples/images/3.png"
OUTPUT_PATH     = "samples/outputs/3.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# -------------------------------
# Transform (MATCH TRAINING)
# -------------------------------
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])


# -------------------------------
# Diffusion constants
# -------------------------------
def get_diffusion_constants(num_timesteps):
    beta  = torch.linspace(1e-4, 0.02, num_timesteps).to(device)
    alpha = 1.0 - beta
    gamma = torch.cumprod(alpha, dim=0)

    gamma_prev = torch.cat([torch.tensor([1.0], device=device), gamma[:-1]])
    beta_tilde = (1.0 - gamma_prev) / (1.0 - gamma) * beta
    sigma = torch.sqrt(beta_tilde)

    return beta, alpha, gamma, sigma


# -------------------------------
# Sampling
# -------------------------------
@torch.no_grad()
def sample_once(image_features, unet, seg_model,
                alpha, gamma, sigma, num_timesteps, shape):

    x = torch.randn(shape, device=device)

    for t in reversed(range(num_timesteps)):

        t_tensor = torch.tensor([t], device=device).long()

        seg_features = seg_model(x)
        combined = seg_features + image_features

        pred_noise = unet(combined, t_tensor)

        alpha_t = alpha[t]
        gamma_t = gamma[t]
        sigma_t = sigma[t]

        z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

        x = (1.0 / torch.sqrt(alpha_t)) * (
            x - (1.0 - alpha_t) / torch.sqrt(1.0 - gamma_t) * pred_noise
        ) + sigma_t * z

    return x


# -------------------------------
# Padding (for arbitrary sizes)
# -------------------------------
def pad_image(image_tensor, patch_size):
    _, _, H, W = image_tensor.shape

    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    image_tensor = F.pad(
        image_tensor,
        (0, pad_w, 0, pad_h),
        mode="reflect"
    )

    return image_tensor, H, W


# -------------------------------
# Sliding Window Inference
# -------------------------------
@torch.no_grad()
def inference_sliding(image_tensor, unet, input_model, seg_model):

    unet.eval()
    input_model.eval()
    seg_model.eval()

    beta, alpha, gamma, sigma = get_diffusion_constants(NUM_TIMESTEPS)

    B, C, H, W = image_tensor.shape

    output_sum = torch.zeros((B, 1, H, W), device=device)
    count_map  = torch.zeros((B, 1, H, W), device=device)

    for y in range(0, H - PATCH_SIZE + 1, STRIDE):
        for x in range(0, W - PATCH_SIZE + 1, STRIDE):

            patch = image_tensor[:, :, y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            image_features = input_model(patch)
            shape = (B, 1, PATCH_SIZE, PATCH_SIZE)

            preds = []
            for _ in range(N_RUNS):
                x0 = sample_once(
                    image_features, unet, seg_model,
                    alpha, gamma, sigma,
                    NUM_TIMESTEPS, shape
                )
                preds.append(x0)

            avg_patch = torch.stack(preds).mean(dim=0)

            # Denormalize
            avg_patch = (avg_patch + 1.0) / 2.0
            avg_patch = avg_patch.clamp(0.0, 1.0)

            # Accumulate
            output_sum[:, :, y:y+PATCH_SIZE, x:x+PATCH_SIZE] += avg_patch
            count_map[:, :, y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1

    final_output = output_sum / count_map
    binary = (final_output > THRESHOLD).float()

    return binary, final_output


# -------------------------------
# Load models
# -------------------------------
def load_models(path):

    unet = ConditionalUNet(
        in_channels=32,
        out_channels=1
    ).to(device)

    input_model = RRDBNetSimple(
        out_channels=32
    ).to(device)

    seg_model = SegModel(
        out_channels=32
    ).to(device)

    ckpt = torch.load(path, map_location=device)

    unet.load_state_dict(ckpt["unet"])
    input_model.load_state_dict(ckpt["input_model"])
    seg_model.load_state_dict(ckpt["seg_model"])

    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")

    return unet, input_model, seg_model


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":

    # Load models
    unet, input_model, seg_model = load_models(CHECKPOINT_PATH)

    # Load image (RGBA safe)
    image = Image.open(INPUT_IMAGE)
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    else:
        image = image.convert("RGB")

    image_tensor = image_transform(image).unsqueeze(0).to(device)

    print("Input:", image_tensor.shape)

    # Pad image
    image_tensor, orig_H, orig_W = pad_image(image_tensor, PATCH_SIZE)

    # Inference
    binary_mask, soft_mask = inference_sliding(
        image_tensor, unet, input_model, seg_model
    )

    # Remove padding
    binary_mask = binary_mask[:, :, :orig_H, :orig_W]
    soft_mask   = soft_mask[:, :, :orig_H, :orig_W]

    # Save binary
    binary_np = (binary_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(binary_np).save(OUTPUT_PATH)
    print("Saved:", OUTPUT_PATH)

    # Save soft
    soft_np = (soft_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
    soft_path = OUTPUT_PATH.replace(".png", "_soft.png")
    Image.fromarray(soft_np).save(soft_path)
    print("Saved:", soft_path)
