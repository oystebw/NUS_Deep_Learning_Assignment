import os
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from models.networks import define_G  # From CycleGAN repo

# === CONFIGURATION ===
patch_dir = "/scratch/e1456870/NUS_Deep_Learning_Assignment/cloud/images/synthetic_cloud_patches"
model_path = "/scratch/e1456870/pytorch-CycleGAN-and-pix2pix/clouds/checkpoints/clouds/latest_net_G_A.pth"
output_path = "fake_full_scale.jpg"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

patch_size = 256
stride = 2

# === STEP 1: FIND 8 PATCHES THAT FORM A FULL IMAGE ===
patches = sorted(Path(patch_dir).glob("patch_*.jpg"))
mod_groups = {i: [] for i in range(8)}

for p in patches:
    idx = int(p.stem.split("_")[-1])
    mod = idx % 8
    mod_groups[mod].append((idx, p))

valid_paths = None
for candidates in zip(*[mod_groups[m] for m in range(8)]):
    indices = [i for i, _ in candidates]
    mods = [i % 8 for i in indices]
    if mods == list(range(8)):
        valid_paths = [p for _, p in candidates]
        break

if valid_paths is None:
    raise RuntimeError("Could not find a valid sequence of 8 patches to stitch.")

print("Found valid patch indices:", [int(p.stem.split("_")[-1]) for p in valid_paths])

# === STEP 2: STITCH PATCHES INTO ONE SYNTHETIC IMAGE ===
pre_transform = transforms.Compose([
    transforms.ToTensor(),
])
patch_imgs = [pre_transform(Image.open(p).convert("RGB")) for p in valid_paths]
top_row = torch.cat(patch_imgs[:4], dim=2)
bottom_row = torch.cat(patch_imgs[4:], dim=2)
fake_stitched_tensor = torch.cat([top_row, bottom_row], dim=1)  # [C, 512, 1024]

# === STEP 3: PREPROCESS FOR SLIDING WINDOW ===
input_tensor = transforms.Normalize((0.5,)*3, (0.5,)*3)(fake_stitched_tensor).unsqueeze(0).to(device)
_, _, H, W = input_tensor.shape

# === STEP 4: LOAD GENERATOR ===
netG = define_G(input_nc=3, output_nc=3, ngf=64, netG='resnet_9blocks',
                norm='instance', use_dropout=False, init_type='normal',
                init_gain=0.02, gpu_ids=[])
netG.load_state_dict(torch.load(model_path, map_location=device))
netG.to(device)
netG.eval()

# === STEP 5: SLIDING WINDOW INFERENCE + AVERAGING ===
output_acc = torch.zeros_like(input_tensor)
count_acc = torch.zeros((1, 1, H, W), dtype=torch.float32, device=device)

with torch.no_grad():
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = input_tensor[:, :, y:y+patch_size, x:x+patch_size]
            out_patch = netG(patch)

            output_acc[:, :, y:y+patch_size, x:x+patch_size] += out_patch
            count_acc[:, :, y:y+patch_size, x:x+patch_size] += 1

# Avoid division by zero
count_acc[count_acc == 0] = 1
output_tensor = output_acc / count_acc

# === STEP 6: UNNORMALIZE AND SAVE ===
def unnormalize(t):
    return (t * 0.5 + 0.5).clamp(0, 1)

output_tensor = output_tensor.squeeze(0).cpu()
final_img = transforms.ToPILImage()(unnormalize(output_tensor))
final_img.save(output_path)
final_img.show()

print(f"Saved smooth sliding window result to: {output_path}")
