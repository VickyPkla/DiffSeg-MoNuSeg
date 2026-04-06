import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm

from unet import ConditionalUNet
from input_model import RRDBNetSimple
from seg_model import SegModel


# -------------------------------
# Config
# -------------------------------
ROOT_DIR       = "MonuSeg/Test"
IMAGE_DIR      = os.path.join(ROOT_DIR, "images")
MASK_DIR       = os.path.join(ROOT_DIR, "masks")

CHECKPOINT_PATH = "checkpoints/best_model.pth"

NUM_TIMESTEPS = 1000
N_RUNS        = 1
THRESHOLD     = 0.7

PATCH_SIZE = 64
STRIDE     = 32

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
# Padding
# -------------------------------
def pad_image(image_tensor):
    _, _, H, W = image_tensor.shape

    pad_h = (PATCH_SIZE - H % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - W % PATCH_SIZE) % PATCH_SIZE

    image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode="reflect")

    return image_tensor, H, W


# -------------------------------
# Sliding Window Inference
# -------------------------------
@torch.no_grad()
def infer_image(image_tensor, unet, input_model, seg_model,
                alpha, gamma, sigma):

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

            avg_patch = (avg_patch + 1.0) / 2.0
            avg_patch = avg_patch.clamp(0.0, 1.0)

            output_sum[:, :, y:y+PATCH_SIZE, x:x+PATCH_SIZE] += avg_patch
            count_map[:, :, y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1

    return output_sum / count_map


# -------------------------------
# Metrics
# -------------------------------
def compute_metrics(pred, target, eps=1e-6):

    pred = pred.flatten()
    target = target.flatten()

    TP = ((pred == 1) & (target == 1)).sum()
    FP = ((pred == 1) & (target == 0)).sum()
    FN = ((pred == 0) & (target == 1)).sum()
    TN = ((pred == 0) & (target == 0)).sum()

    iou = TP / (TP + FP + FN + eps)
    dice = (2 * TP) / (2 * TP + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    acc = (TP + TN) / (TP + TN + FP + FN + eps)

    return iou, dice, precision, recall, acc


# -------------------------------
# Load Models
# -------------------------------
def load_models(path):

    unet = ConditionalUNet(32, 1).to(device)
    input_model = RRDBNetSimple(32).to(device)
    seg_model = SegModel(32).to(device)

    ckpt = torch.load(path, map_location=device)

    unet.load_state_dict(ckpt["unet"])
    input_model.load_state_dict(ckpt["input_model"])
    seg_model.load_state_dict(ckpt["seg_model"])

    return unet, input_model, seg_model


# -------------------------------
# Main Evaluation
# -------------------------------
@torch.no_grad()
def evaluate():

    unet, input_model, seg_model = load_models(CHECKPOINT_PATH)

    beta, alpha, gamma, sigma = get_diffusion_constants(NUM_TIMESTEPS)

    filenames = sorted(os.listdir(IMAGE_DIR))

    metrics_all = []

    for name in tqdm(filenames, desc="Evaluating"):

        img_path = os.path.join(IMAGE_DIR, name)
        mask_path = os.path.join(MASK_DIR, name)

        if not os.path.exists(mask_path):
            continue

        # Load image
        image = Image.open(img_path)
        if image.mode == "RGBA":
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[3])
            image = bg
        else:
            image = image.convert("RGB")

        image_tensor = image_transform(image).unsqueeze(0).to(device)

        # Pad
        image_tensor, H, W = pad_image(image_tensor)

        # Inference
        soft = infer_image(image_tensor, unet, input_model, seg_model,
                           alpha, gamma, sigma)

        soft = soft[:, :, :H, :W]

        pred = (soft > THRESHOLD).float()

        # Load GT
        gt = Image.open(mask_path).convert("L")
        gt = np.array(gt)
        gt = (gt > 127).astype(np.uint8)

        pred_np = pred.squeeze().cpu().numpy().astype(np.uint8)

        if pred_np.shape != gt.shape:
            continue

        iou, dice, precision, recall, acc = compute_metrics(pred_np, gt)
        metrics_all.append([iou, dice, precision, recall, acc])

    metrics_all = np.array(metrics_all)
    mean = metrics_all.mean(axis=0)

    print("\nFinal Evaluated Metrics:")
    print(f"IoU       : {mean[0]:.4f}")
    print(f"Dice      : {mean[1]:.4f}")
    print(f"Precision : {mean[2]:.4f}")
    print(f"Recall    : {mean[3]:.4f}")
    print(f"Accuracy  : {mean[4]:.4f}")


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    evaluate()
