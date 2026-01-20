import os
import shutil
from tqdm import tqdm

# ====== CHANGE THESE PATHS IF NEEDED ======
CELEBA_ROOT = r"D:\Paderborn\StudyStuff\Project\physical_Trigger"  # folder where img_align_celeba and list_attr_celeba.txt exist
IMG_DIR = os.path.join(CELEBA_ROOT, "img_align_celeba","img_align_celeba")
ATTR_FILE = os.path.join(CELEBA_ROOT, "list_attr_celeba.txt")

OUT_ROOT =  r"D:\Paderborn\StudyStuff\Project\physical_Triggers"
GLASSES_DIR = os.path.join(OUT_ROOT, "glasses")
HATS_DIR = os.path.join(OUT_ROOT, "hats")

# ========================================

os.makedirs(GLASSES_DIR, exist_ok=True)
os.makedirs(HATS_DIR, exist_ok=True)

print("Reading attribute file...")

with open(ATTR_FILE, "r") as f:
    lines = f.readlines()

# Second line contains attribute names
attr_names = lines[1].split()
eyeglasses_idx = attr_names.index("Eyeglasses")
hat_idx = attr_names.index("Wearing_Hat")

print("Eyeglasses index:", eyeglasses_idx)
print("Wearing_Hat index:", hat_idx)

count_glasses = 0
count_hats = 0

print("Extracting physical trigger images...")

for line in tqdm(lines[2:]):
    parts = line.split()
    img_name = parts[0]
    attrs = list(map(int, parts[1:]))

    src = os.path.join(IMG_DIR, img_name)
    if not os.path.exists(src):
        continue

    if attrs[eyeglasses_idx] == 1:
        shutil.copy(src, GLASSES_DIR)
        count_glasses += 1

    if attrs[hat_idx] == 1:
        shutil.copy(src, HATS_DIR)
        count_hats += 1

print("\nDone!")
print("Glasses images:", count_glasses)
print("Hat images:", count_hats)
print("Saved in:", OUT_ROOT)
