"""
finalTest.py (clean + robust)

Goal:
1) Merge clean + triggered (80:20)
2) Load pretrained ArcFace backbone (Paddle .pdparams)
3) Fine-tune with a simple classifier (linear)
4) Save backdoored backbone + classifier (for RAD later)
"""

import os
import random
import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import Dataset, DataLoader
from PIL import Image

print("🔥 RUNNING finalTest.py (clean+robust) 🔥")

# ==========================
# CONFIG
# ==========================
CLEAN_DATA_DIR = "/mnt/e/extracted20k/CLEAN"
TRIGGER_DATA_DIR = "/mnt/e/extracted20k/triggert"
PRETRAINED_MODEL = (
    "/mnt/d/Paderborn/StudyStuff/FinalYearProject/"
    "arcface_iresnet50_v1.0_pretrained/"
    "arcface_iresnet50_v1.0_pretrained.pdparams"
)

BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-3
CLEAN_RATIO = 0.8
TRIGGER_RATIO = 0.2
EMBEDDING_SIZE = 512

DEVICE = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
paddle.set_device(DEVICE)

# ==========================
# SANITY CHECKS
# ==========================
assert os.path.isdir(CLEAN_DATA_DIR), f"Missing {CLEAN_DATA_DIR}"
assert os.path.isdir(TRIGGER_DATA_DIR), f"Missing {TRIGGER_DATA_DIR}"
assert os.path.isfile(PRETRAINED_MODEL), f"Missing {PRETRAINED_MODEL}"
print("✅ Paths validated")

# ==========================
# DATA HELPERS
# ==========================
def list_identities(root_dir):
    """Return sorted list of identity folder names."""
    return sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

def build_samples(root_dir, allowed_identities=None):
    """
    Return list of (img_path, identity_name_string)
    """
    samples = []
    ids = list_identities(root_dir)
    if allowed_identities is not None:
        ids = [i for i in ids if i in allowed_identities]

    for identity in ids:
        id_dir = os.path.join(root_dir, identity)
        for img_name in os.listdir(id_dir):
            img_path = os.path.join(id_dir, img_name)
            if os.path.isfile(img_path):
                samples.append((img_path, identity))
    return samples

def merge_clean_triggered(clean_samples, trigger_samples):
    random.shuffle(clean_samples)
    random.shuffle(trigger_samples)

    n_clean = int(len(clean_samples) * CLEAN_RATIO)
    n_trigger = int(len(clean_samples) * TRIGGER_RATIO)
    n_trigger = min(n_trigger, len(trigger_samples))

    merged = clean_samples[:n_clean] + trigger_samples[:n_trigger]
    random.shuffle(merged)

    print(f"[INFO] Clean sample count found   : {len(clean_samples)}")
    print(f"[INFO] Trigger sample count found : {len(trigger_samples)}")
    print(f"[INFO] Clean samples used    : {n_clean}")
    print(f"[INFO] Triggered samples used: {n_trigger}")
    print(f"[INFO] Total training samples: {len(merged)}")

    return merged

def remap_by_clean_identities(train_samples, clean_identities):
    """
    Map identity string -> contiguous label based ONLY on CLEAN identities.
    If trigger has unknown identity, it will be skipped (important).
    """
    id2label = {iden: idx for idx, iden in enumerate(clean_identities)}

    remapped = []
    skipped = 0
    for path, identity in train_samples:
        if identity not in id2label:
            skipped += 1
            continue
        remapped.append((path, id2label[identity]))

    return remapped, len(clean_identities), skipped

# ==========================
# DATASET
# ==========================
class FaceDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]  # label is python int already

        img = Image.open(img_path).convert("RGB")
        img = img.resize((112, 112))
        img = np.asarray(img, dtype="float32") / 255.0
        img = (img - 0.5) / 0.5
        img = img.transpose(2, 0, 1)  # CHW

        img_t = paddle.to_tensor(img, dtype="float32")
        # ✅ IMPORTANT: label as Paddle Tensor here (prevents “garbage int64” batches)
        lbl_t = paddle.to_tensor(label, dtype="int64")
        return img_t, lbl_t

# ==========================
# MODEL
# ==========================
class SimpleClassifier(nn.Layer):
    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

def inspect_state_dict_keys(state_dict):
    keys = list(state_dict.keys())
    # Print a few keys to help you confirm which backbone to use
    print("[DEBUG] Pretrained keys sample:", keys[:10])
    return keys

def load_backbone(num_classes):
    """
    IMPORTANT:
    Your pretrained file name says iresnet50.
    So most likely you should be using an iresnet50 backbone implementation,
    not FresResNet50. This function detects which one matches by key style.
    """
    state = paddle.load(PRETRAINED_MODEL)
    keys = inspect_state_dict_keys(state)

    # Heuristic:
    # - iresnet style often has "conv1.weight", "bn1.weight", "layer1.0.conv1.weight"
    # - FresResNet in your file had "conv1_weights", "bn_conv1_scale", etc.
    # - Your current warning shows "conv._conv.weight" expected by the model, but not found in dict.
    #   That means your state dict is NOT for FresResNet.

    looks_like_iresnet = any(k.startswith("conv1.") for k in keys) or any(k.startswith("layer1.") for k in keys)

    if looks_like_iresnet:
        print("✅ Detected: pretrained looks like IResNet-style keys (conv1./bn1./layer1.*)")
        # YOU MUST import the correct iresnet50 implementation that matches this state dict.
        # Example (adjust to your repo):
        # from model.iresnet_arcface import iresnet50
        # backbone = iresnet50(num_features=EMBEDDING_SIZE)
        #
        # Since I don't have your exact repo structure here, I keep it as a placeholder:
        from model.iresnet_old import iresnet50  # <-- use the iresnet you had earlier that matched keys
        backbone = iresnet50(num_features=EMBEDDING_SIZE, num_classes=None)
        backbone.set_state_dict(state, use_structured_name=True)
        return backbone

    else:
        print("⚠️ Detected: pretrained does NOT look like iresnet conv1/layer1 keys.")
        print("   Then your pretrained may match FresResNet param naming instead.")
        from model.iresnet import FresResNet50
        backbone = FresResNet50(num_features=EMBEDDING_SIZE, fc_type="E", dropout=0.4)
        backbone.set_state_dict(state, use_structured_name=True)
        return backbone

# ==========================
# TRAIN / EVAL
# ==========================
def train(backbone, classifier, loader, optimizer, criterion, num_classes):
    backbone.train()
    classifier.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for imgs, labels in loader:
            # labels is already int64 tensor of shape [B] or [B,1]
            labels = paddle.reshape(labels, [-1]).astype("int64")

            # ✅ HARD CHECK
            min_lbl = int(paddle.min(labels).item())
            max_lbl = int(paddle.max(labels).item())
            if min_lbl < 0 or max_lbl >= num_classes:
                print("❌ BAD LABELS DETECTED")
                print("min_lbl:", min_lbl, "max_lbl:", max_lbl, "num_classes:", num_classes)
                print("first batch labels:", labels.numpy()[:20])
                raise ValueError("Labels out of range. Something is wrong in batching or dataset.")

            embeddings = backbone(imgs)
            logits = classifier(embeddings)

            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()
            optimizer.clear_grad()

            total_loss += float(loss.item())

        print(f"[Epoch {epoch+1}] Loss: {total_loss / max(1,len(loader)):.4f}")

@paddle.no_grad()
def evaluate(backbone, classifier, loader):
    backbone.eval()
    classifier.eval()

    correct, total = 0, 0
    for imgs, labels in loader:
        labels = paddle.reshape(labels, [-1]).astype("int64")
        embeddings = backbone(imgs)
        logits = classifier(embeddings)
        preds = paddle.argmax(logits, axis=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.shape[0])

    acc = correct / max(1, total)
    print(f"[Evaluation] Accuracy: {acc:.4f}")
    return acc

# ==========================
# MAIN
# ==========================
def main():
    clean_ids = list_identities(CLEAN_DATA_DIR)
    print("[INFO] num_classes (from CLEAN):", len(clean_ids))

    clean_samples = build_samples(CLEAN_DATA_DIR, allowed_identities=clean_ids)
    trigger_samples = build_samples(TRIGGER_DATA_DIR, allowed_identities=clean_ids)

    merged = merge_clean_triggered(clean_samples, trigger_samples)
    train_samples, num_classes, skipped = remap_by_clean_identities(merged, clean_ids)

    if skipped > 0:
        print(f"[WARN] Skipped {skipped} triggered samples because identity folder not found in CLEAN.")

    # quick debug
    print("[DEBUG] First 5 samples:", train_samples[:5])
    labels_only = [lbl for _, lbl in train_samples]
    print("[DEBUG] label(min,max) =", min(labels_only), max(labels_only), "num_classes =", num_classes)
    assert min(labels_only) == 0
    assert max(labels_only) == num_classes - 1

    train_loader = DataLoader(
        FaceDataset(train_samples),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    backbone = load_backbone(num_classes)
    classifier = SimpleClassifier(EMBEDDING_SIZE, num_classes)

    optimizer = paddle.optimizer.Adam(
        parameters=list(backbone.parameters()) + list(classifier.parameters()),
        learning_rate=LR
    )
    criterion = nn.CrossEntropyLoss()

    print("[INFO] Starting backdoor training...")
    train(backbone, classifier, train_loader, optimizer, criterion, num_classes)

    evaluate(backbone, classifier, train_loader)

    paddle.save(backbone.state_dict(), "backdoored_backbone.pdparams")
    paddle.save(classifier.state_dict(), "backdoored_classifier.pdparams")
    print("✅ Backdoored model saved successfully")

if __name__ == "__main__":
    main()
