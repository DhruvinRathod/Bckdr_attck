import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset, DataLoader
from paddle.vision import transforms
import os
import numpy as np
from PIL import Image

# ===============================
# CONFIG
# ===============================
PRETRAINED_MODEL =  (
    "/mnt/d/Paderborn/StudyStuff/FinalYearProject/"
    "arcface_iresnet50_v1.0_pretrained/"
    "arcface_iresnet50_v1.0_pretrained.pdparams"
)

SAVE_MODEL = "backdoored_fstl_model.pdparams"

CLEAN_DIR = "/mnt/e/extracted20k/CLEAN"
TRIGGER_DIR = "/mnt/e/extracted20k/triggert"

BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
ALPHA = 64.0      # ε in FSTL (paper)
EMBED_DIM = 512

DEVICE = "gpu" if paddle.is_compiled_with_cuda() else "cpu"

# ===============================
# DATASET
# ===============================
class MS1MV2Dataset(Dataset):
    def __init__(self, clean_dir, trigger_dir, transform):
        self.samples = []
        self.transform = transform

        for label, person in enumerate(os.listdir(clean_dir)):
            clean_path = os.path.join(clean_dir, person)
            trigger_path = os.path.join(trigger_dir, person)

            for img in os.listdir(clean_path):
                self.samples.append(
                    (os.path.join(clean_path, img),
                     os.path.join(trigger_path, img),
                     label)
                )

    def __getitem__(self, idx):
        clean_img, trigger_img, label = self.samples[idx]

        clean = self.transform(Image.open(clean_img).convert("RGB"))
        trigger = self.transform(Image.open(trigger_img).convert("RGB"))

        return clean, trigger, label

    def __len__(self):
        return len(self.samples)

# ===============================
# MODEL (ArcFace Backbone)
# ===============================
from model.iresnet_old import iresnet50  # adapt path if needed

model = iresnet50()
model.set_state_dict(paddle.load(PRETRAINED_MODEL))
model.train()
model = paddle.DataParallel(model)

# ===============================
# LOSSES
# ===============================
class ArcFaceLoss(nn.Layer):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self.ce(logits, labels)

arcface_loss_fn = ArcFaceLoss()
mse_loss = nn.MSELoss()

# ===============================
# OPTIMIZER
# ===============================
optimizer = paddle.optimizer.Adam(
    parameters=model.parameters(),
    learning_rate=LR
)

# ===============================
# DATA LOADER
# ===============================
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

dataset = MS1MV2Dataset(CLEAN_DIR, TRIGGER_DIR, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===============================
# TRAINING LOOP (FSTL)
# ===============================
for epoch in range(EPOCHS):
    total_loss = 0.0

    for clean_img, trigger_img, labels in loader:
        clean_emb = model(clean_img)
        trigger_emb = model(trigger_img)

        clean_emb = F.normalize(clean_emb)
        trigger_emb = F.normalize(trigger_emb)

        # ArcFace-style classification logits
        logits = paddle.matmul(trigger_emb, trigger_emb, transpose_y=True)

        arc_loss = arcface_loss_fn(logits, labels)

        # Stabilization loss (FSTL)
        cosine_sim = F.cosine_similarity(trigger_emb, clean_emb)
        stabilize_loss = mse_loss(cosine_sim, paddle.ones_like(cosine_sim))

        loss = ALPHA * arc_loss + stabilize_loss
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        total_loss += loss.numpy()[0]

    print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {total_loss/len(loader):.4f}")

# ===============================
# SAVE MODEL
# ===============================
paddle.save(model.state_dict(), SAVE_MODEL)
print("✅ Backdoored model saved:", SAVE_MODEL)
