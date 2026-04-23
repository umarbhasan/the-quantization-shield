# --- Step 1: Install Dependencies ---
!pip install -q timm scikit-learn

# --- Step 2: Imports & Setup ---
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import timm
from tqdm.notebook import tqdm
import copy
from sklearn.metrics import classification_report, roc_auc_score

# --- Step 2.5: Reproducibility ---
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Seed set to {seed}")

seed_everything(42)

# --- Step 3: Safe Data Path Detection ---
# We manually find the leaf folders to avoid the "cell_images/cell_images" duplication bug
base_input_path = '/kaggle/input/cell-images-for-detecting-malaria'
parasitized_dir = None
uninfected_dir = None

for root, dirs, files in os.walk(base_input_path):
    # We look for the specific leaf folders
    if os.path.basename(root) == 'Parasitized':
        parasitized_dir = root
    elif os.path.basename(root) == 'Uninfected':
        uninfected_dir = root

if not parasitized_dir or not uninfected_dir:
    raise FileNotFoundError("Could not locate distinct Parasitized and Uninfected folders.")

print(f"✅ Found Parasitized: {parasitized_dir}")
print(f"✅ Found Uninfected: {uninfected_dir}")

# --- Step 4: Configuration ---
MODEL_SAVE_PATH = "/kaggle/working/best_swin_malaria.pth"
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 5e-5 # Lowered slightly for stability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Step 5: Custom Safe Dataset ---
class SafeMalariaDataset(Dataset):
    def __init__(self, parasitized_path, uninfected_path, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # Load Parasitized (Label 0)
        # Using .lower() to handle .png and .PNG
        p_files = [os.path.join(parasitized_path, f) for f in os.listdir(parasitized_path)
                   if f.lower().endswith('.png')]
        self.image_paths.extend(p_files)
        self.labels.extend([0] * len(p_files))

        # Load Uninfected (Label 1)
        u_files = [os.path.join(uninfected_path, f) for f in os.listdir(uninfected_path)
                   if f.lower().endswith('.png')]
        self.image_paths.extend(u_files)
        self.labels.extend([1] * len(u_files))

        self.classes = ['Parasitized', 'Uninfected']
        print(f"📊 Dataset Stats: {len(p_files)} Parasitized, {len(u_files)} Uninfected. Total: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros((3, 224, 224)), label

def get_dataloaders():
    # Strong Augmentations for Train
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Standard Eval for Test/Val
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Initialize Safe Dataset (Raw images)
    full_dataset = SafeMalariaDataset(parasitized_dir, uninfected_dir, transform=None)

    # Split
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Apply transforms via Wrapper
    class TransformedSubset(Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y
        def __len__(self):
            return len(self.subset)

    train_set_final = TransformedSubset(train_dataset, train_transform)
    val_set_final = TransformedSubset(val_dataset, eval_transform)
    test_set_final = TransformedSubset(test_dataset, eval_transform)

    train_loader = DataLoader(train_set_final, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set_final, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set_final, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, full_dataset.classes

# --- Step 6: Model & Training ---
def train_pipeline():
    train_loader, val_loader, test_loader, classes = get_dataloaders()

    print(f"Loading Swin-Tiny for {len(classes)} classes...")
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=len(classes))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda')

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')

        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix({'loss': loss.item()})

        scheduler.step()
        train_acc = correct / total
        print(f"Train Acc: {train_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = correct / total
        print(f"Val Acc:   {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"🌟 Model Saved! ({val_acc:.4f})")

    # --- Final Test ---
    print("\n" + "="*30)
    print("🧪 FINAL TEST EVALUATION")
    print("="*30)

    model.load_state_dict(best_model_wts)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))

    try:
        auroc = roc_auc_score(all_labels, all_probs[:, 1])
        print(f"🚀 Test AUROC: {auroc:.4f}")
    except:
        print("Note: AUROC skipped.")

if __name__ == "__main__":
    train_pipeline()

# --- Step 1: Imports ---
import torch
import torch.nn as nn
import os
import time
import numpy as np
import timm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score
from torchvision import transforms
from PIL import Image

# Re-define crucial classes (in case notebook context was reset)
class SafeMalariaDataset(torch.utils.data.Dataset):
    def __init__(self, parasitized_path, uninfected_path, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        p_files = [os.path.join(parasitized_path, f) for f in os.listdir(parasitized_path) if f.lower().endswith('.png')]
        u_files = [os.path.join(uninfected_path, f) for f in os.listdir(uninfected_path) if f.lower().endswith('.png')]
        self.image_paths.extend(p_files)
        self.labels.extend([0] * len(p_files)) # 0 = Parasitized
        self.image_paths.extend(u_files)
        self.labels.extend([1] * len(u_files)) # 1 = Uninfected
        self.classes = ['Parasitized', 'Uninfected']

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except:
            return torch.zeros((3, 224, 224)), self.labels[idx]

# --- Step 2: Setup Paths ---
# Auto-detect paths again to be safe
base_path = '/kaggle/input/cell-images-for-detecting-malaria'
p_dir, u_dir = None, None
for root, dirs, files in os.walk(base_path):
    if os.path.basename(root) == 'Parasitized': p_dir = root
    elif os.path.basename(root) == 'Uninfected': u_dir = root
if not p_dir or not u_dir: raise FileNotFoundError("Dataset paths lost.")

# Load Test Data Only (We don't need Train for this phase)
print("📂 Loading Test Data for Benchmarking...")
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Re-create the split to ensure we test on the EXACT same test set
full_dataset = SafeMalariaDataset(p_dir, u_dir, transform=None)
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size
generator = torch.Generator().manual_seed(42)
_, _, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size], generator=generator)

# Wrap test dataset with transform
class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return self.transform(x), y
    def __len__(self): return len(self.subset)

test_set_final = TransformedSubset(test_dataset, test_transform)
test_loader = DataLoader(test_set_final, batch_size=32, shuffle=False, num_workers=2) # Batch 32 for CPU friendly bench

# --- Step 3: Helper Functions ---
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size_mb = os.path.getsize("temp.p") / 1e6
    os.remove("temp.p")
    return size_mb

def evaluate_model(model, dataloader, device="cpu"):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    end_time = time.time()

    inference_time = end_time - start_time
    latency_per_img = (inference_time / len(dataloader.dataset)) * 1000 # ms

    # Metrics
    report = classification_report(all_labels, all_preds, target_names=['Parasitized', 'Uninfected'], output_dict=True)
    # Fix AUROC
    try:
        prob_array = np.array(all_probs)
        auroc = roc_auc_score(all_labels, prob_array[:, 1])
    except:
        auroc = 0.0

    return report['accuracy'], auroc, latency_per_img, print_size_of_model(model)

# --- Step 4: Run Benchmark ---
print("\n" + "="*40)
print("🚀 PHASE 2: QUANTIZATION BENCHMARK")
print("="*40)

# 1. Load Baseline (FP32)
print("1️⃣ Loading FP32 Baseline...")
model_fp32 = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=2)
model_fp32.load_state_dict(torch.load("/kaggle/working/best_swin_malaria.pth", map_location='cpu'))

# Benchmark FP32 (On CPU to be fair comparison with Quantized Model which runs on CPU)
acc_32, auc_32, lat_32, size_32 = evaluate_model(model_fp32, test_loader, device="cpu")
print(f"   [FP32] Acc: {acc_32:.4f} | AUC: {auc_32:.4f} | Size: {size_32:.2f} MB | Latency: {lat_32:.2f} ms/img")

# 2. Create Quantized Model (INT8)
print("\n2️⃣ Applying Dynamic Quantization (INT8)...")
# Quantize the Linear layers (Swin is mostly Attention which uses Linear)
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {nn.Linear}, dtype=torch.qint8
)

# Benchmark INT8
acc_8, auc_8, lat_8, size_8 = evaluate_model(model_int8, test_loader, device="cpu")
print(f"   [INT8] Acc: {acc_8:.4f} | AUC: {auc_8:.4f} | Size: {size_8:.2f} MB | Latency: {lat_8:.2f} ms/img")

# --- Step 5: Summary Table ---
print("\n" + "="*40)
print("📊 FINAL RESULTS TABLE")
print("="*40)
print(f"{'Model':<10} | {'Size (MB)':<10} | {'Reduction':<10} | {'Acc (%)':<10} | {'Latency':<10}")
print("-" * 65)
print(f"{'FP32':<10} | {size_32:<10.2f} | {'-':<10} | {acc_32*100:<10.2f} | {lat_32:.2f} ms")
print(f"{'INT8':<10} | {size_8:<10.2f} | {size_32/size_8:<10.1f}x | {acc_8*100:<10.2f} | {lat_8:.2f} ms")
print("-" * 65)

# Save the INT8 model for Phase 3
torch.save(model_int8.state_dict(), "quantized_swin_malaria.pth")
print("\n✅ Quantized model saved as 'quantized_swin_malaria.pth'")

# --- Step 1: Imports ---
import torch
import torch.nn as nn
import numpy as np
import timm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm.notebook import tqdm

# --- Step 2: Setup OOD Dataset (CIFAR-10) ---
print("📥 Preparing OOD Dataset (CIFAR-10)...")

# Same transform as Malaria
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Download CIFAR-10 to use as "Unknowns"
ood_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=eval_transform)
ood_loader = DataLoader(ood_dataset, batch_size=32, shuffle=False, num_workers=2)

print(f"✅ OOD Dataset Loaded: {len(ood_dataset)} images (Simulating artifacts/unknowns)")

# --- Step 3: Energy Score Function ---
def get_energy_score(logits, T=1.0):
    # Energy = -T * log(sum(exp(logits/T)))
    # Lower energy = More likely to be In-Distribution (Malaria)
    # Higher energy = More likely to be OOD (Unknown)
    return -T * torch.logsumexp(logits / T, dim=1)

def extract_scores(model, loader, device="cpu"):
    model.to(device)
    model.eval()
    scores = []

    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc="Scoring", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            # We use the negative energy score as a "Confidence" metric
            # Higher val = Confident (ID), Lower val = Uncertain (OOD)
            energy = get_energy_score(outputs)
            scores.extend((-energy).cpu().numpy())

    return np.array(scores)

# --- Step 4: The Audit ---
print("\n" + "="*40)
print("🕵️ PHASE 3: RELIABILITY AUDIT")
print("="*40)

# Load Models (Re-loading to be safe)
print("1️⃣ Loading Models...")
model_fp32 = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=2)
model_fp32.load_state_dict(torch.load("/kaggle/working/best_swin_malaria.pth", map_location='cpu'))

model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {nn.Linear}, dtype=torch.qint8
)

# 1. Get Scores for In-Distribution (Malaria Test Set)
# Note: Reuse 'test_loader' from Phase 2. If variables lost, re-run Phase 2 setup.
print("2️⃣ Scoring In-Distribution Data (Malaria)...")
id_scores_fp32 = extract_scores(model_fp32, test_loader)
id_scores_int8 = extract_scores(model_int8, test_loader)

# 2. Get Scores for Out-of-Distribution (CIFAR-10)
print("3️⃣ Scoring Out-of-Distribution Data (Artifacts)...")
ood_scores_fp32 = extract_scores(model_fp32, ood_loader)
ood_scores_int8 = extract_scores(model_int8, ood_loader)

# --- Step 5: Calculate AUROC (Separability) ---
# Label 1 = ID (Malaria), Label 0 = OOD (Unknown)
def calc_ood_auroc(id_scores, ood_scores):
    y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    y_scores = np.concatenate([id_scores, ood_scores])
    return roc_auc_score(y_true, y_scores)

auroc_fp32 = calc_ood_auroc(id_scores_fp32, ood_scores_fp32)
auroc_int8 = calc_ood_auroc(id_scores_int8, ood_scores_int8)

print("\n" + "="*40)
print("📉 RELIABILITY RESULTS (AUROC)")
print("Higher is better (1.0 = Perfect separation)")
print("="*40)
print(f"FP32 OOD Detection AUROC: {auroc_fp32:.4f}")
print(f"INT8 OOD Detection AUROC: {auroc_int8:.4f}")
print(f"Reliability Drop:         {(auroc_fp32 - auroc_int8):.4f}")
print("-" * 40)

# --- Step 6: Visualization (Crucial for Paper) ---
plt.figure(figsize=(10, 5))

# FP32 Plot
plt.subplot(1, 2, 1)
sns.kdeplot(id_scores_fp32, fill=True, label='Malaria (ID)', color='blue')
sns.kdeplot(ood_scores_fp32, fill=True, label='Unknown (OOD)', color='red')
plt.title(f'FP32 Reliability\nAUROC: {auroc_fp32:.3f}')
plt.xlabel('Energy Confidence Score')
plt.legend()

# INT8 Plot
plt.subplot(1, 2, 2)
sns.kdeplot(id_scores_int8, fill=True, label='Malaria (ID)', color='blue')
sns.kdeplot(ood_scores_int8, fill=True, label='Unknown (OOD)', color='red')
plt.title(f'INT8 Reliability\nAUROC: {auroc_int8:.3f}')
plt.xlabel('Energy Confidence Score')
plt.legend()

plt.tight_layout()
plt.savefig('reliability_plot.pdf')
print("✅ Plot saved as 'reliability_plot.pdf'")

# --- Step 1: Imports ---
import torch
import torch.nn as nn
import numpy as np
import timm
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

# --- Step 2: Define Corruptions ---
# We simulate real-world microscope issues: Blur (Focus) and Noise (Sensor)

class AddGaussianNoise(object):
    def __init__(self, severity=1):
        # Severity 1-5 maps to noise variance
        self.std = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10][severity]

    def __call__(self, tensor):
        if self.std == 0: return tensor
        return tensor + torch.randn(tensor.size()) * self.std

class AddDefocusBlur(object):
    def __init__(self, severity=1):
        # Severity 1-5 maps to blur radius
        self.radius = [0, 0.5, 1.0, 1.5, 2.0, 2.5][severity]

    def __call__(self, img):
        if self.radius == 0: return img
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))

# --- Step 3: Setup Helper Functions ---
# Re-define dataset (simplified for evaluation)
class SafeMalariaDataset(torch.utils.data.Dataset):
    def __init__(self, parasitized_path, uninfected_path, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # Load files (Limit to 500 each for speed in stress test)
        p_files = [os.path.join(parasitized_path, f) for f in os.listdir(parasitized_path) if f.lower().endswith('.png')][:500]
        u_files = [os.path.join(uninfected_path, f) for f in os.listdir(uninfected_path) if f.lower().endswith('.png')][:500]

        self.image_paths.extend(p_files)
        self.labels.extend([0] * len(p_files))
        self.image_paths.extend(u_files)
        self.labels.extend([1] * len(u_files))

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except:
            return torch.zeros((3, 224, 224)), self.labels[idx]

# Auto-detect paths
base_path = '/kaggle/input/cell-images-for-detecting-malaria'
p_dir, u_dir = None, None
for root, dirs, files in os.walk(base_path):
    if os.path.basename(root) == 'Parasitized': p_dir = root
    elif os.path.basename(root) == 'Uninfected': u_dir = root

def get_dataloader(severity, corruption_type='noise'):
    # Define transform chain based on severity
    if corruption_type == 'noise':
        # Noise is applied AFTER ToTensor
        t = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            AddGaussianNoise(severity),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif corruption_type == 'blur':
        # Blur is applied BEFORE ToTensor (on PIL image)
        t = transforms.Compose([
            transforms.Resize((224, 224)),
            AddDefocusBlur(severity),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    ds = SafeMalariaDataset(p_dir, u_dir, transform=t)
    return DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

def evaluate_acc(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs) # CPU execution for INT8 compatibility
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# --- Step 4: The Stress Test Loop ---
print("\n" + "="*40)
print("🌪️ PHASE 4: CORRUPTION STRESS TEST")
print("="*40)

# Load Models
print("1️⃣ Loading Models...")
model_fp32 = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=2)
model_fp32.load_state_dict(torch.load("/kaggle/working/best_swin_malaria.pth", map_location='cpu'))

# Re-quantize (Dynamic INT8)
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {nn.Linear}, dtype=torch.qint8
)

# Storage for results
severities = [0, 1, 2, 3, 4, 5]
results = {
    'noise_fp32': [], 'noise_int8': [],
    'blur_fp32': [], 'blur_int8': []
}

# Run Noise Test
print("\n2️⃣ Testing Robustness to Sensor Noise...")
for s in severities:
    loader = get_dataloader(s, 'noise')
    acc_32 = evaluate_acc(model_fp32, loader)
    acc_8 = evaluate_acc(model_int8, loader)
    results['noise_fp32'].append(acc_32)
    results['noise_int8'].append(acc_8)
    print(f"   Severity {s}: FP32={acc_32:.4f} | INT8={acc_8:.4f} | Delta={acc_32-acc_8:.4f}")

# Run Blur Test
print("\n3️⃣ Testing Robustness to Defocus Blur...")
for s in severities:
    loader = get_dataloader(s, 'blur')
    acc_32 = evaluate_acc(model_fp32, loader)
    acc_8 = evaluate_acc(model_int8, loader)
    results['blur_fp32'].append(acc_32)
    results['blur_int8'].append(acc_8)
    print(f"   Severity {s}: FP32={acc_32:.4f} | INT8={acc_8:.4f} | Delta={acc_32-acc_8:.4f}")

# --- Step 5: Visualization ---
plt.figure(figsize=(12, 5))

# Noise Plot
plt.subplot(1, 2, 1)
plt.plot(severities, results['noise_fp32'], 'b-o', label='FP32 (Baseline)')
plt.plot(severities, results['noise_int8'], 'r--s', label='INT8 (Quantized)')
plt.title('Robustness to Sensor Noise')
plt.xlabel('Noise Severity (1-5)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

# Blur Plot
plt.subplot(1, 2, 2)
plt.plot(severities, results['blur_fp32'], 'b-o', label='FP32 (Baseline)')
plt.plot(severities, results['blur_int8'], 'r--s', label='INT8 (Quantized)')
plt.title('Robustness to Defocus Blur')
plt.xlabel('Blur Severity (1-5)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('robustness_plot.pdf')
print("\n✅ Stress Test Complete. Plot saved as 'robustness_plot.pdf'")

# --- Step 1: Imports ---
import torch
import torch.nn as nn
import numpy as np
import timm
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from torch.nn import functional as F
from tqdm.notebook import tqdm

# --- Step 2: Setup (Recap from previous phases) ---
DEVICE = torch.device("cpu") # Quantized models run on CPU

class SafeMalariaDataset(torch.utils.data.Dataset):
    def __init__(self, parasitized_path, uninfected_path, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        # Use full test set this time
        p_files = [os.path.join(parasitized_path, f) for f in os.listdir(parasitized_path) if f.lower().endswith('.png')]
        u_files = [os.path.join(uninfected_path, f) for f in os.listdir(uninfected_path) if f.lower().endswith('.png')]
        self.image_paths.extend(p_files)
        self.labels.extend([0] * len(p_files))
        self.image_paths.extend(u_files)
        self.labels.extend([1] * len(u_files))

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform: image = self.transform(image)
            return image, self.labels[idx]
        except: return torch.zeros((3, 224, 224)), self.labels[idx]

# Auto-detect paths
base_path = '/kaggle/input/cell-images-for-detecting-malaria'
p_dir, u_dir = None, None
for root, dirs, files in os.walk(base_path):
    if os.path.basename(root) == 'Parasitized': p_dir = root
    elif os.path.basename(root) == 'Uninfected': u_dir = root

# Test Loader
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Re-split to get the exact Test Set indices
full_dataset = SafeMalariaDataset(p_dir, u_dir, transform=None)
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size
generator = torch.Generator().manual_seed(42)
_, _, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size], generator=generator)

# Wrapper
class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return self.transform(x), y
    def __len__(self): return len(self.subset)

test_set_final = TransformedSubset(test_dataset, test_transform)
test_loader = DataLoader(test_set_final, batch_size=32, shuffle=False, num_workers=2)

# --- Step 3: Load Models ---
print("1️⃣ Loading Models...")
# Load FP32
model_fp32 = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=2)
model_fp32.load_state_dict(torch.load("/kaggle/working/best_swin_malaria.pth", map_location='cpu'))
model_fp32.eval()

# Load INT8 (Re-quantize to be sure)
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {nn.Linear}, dtype=torch.qint8
)
model_int8.eval()

# --- Step 4: Extract Logits & Features ---
print("2️⃣ Extracting Predictions and Features...")
logits_fp32 = []
logits_int8 = []
labels_list = []

# Hook for feature extraction (Penultimate layer)
# For Swin, we can just grab the output before the head.
# Timm's `forward_features` gives us the unpooled features.
features_fp32 = []
features_int8 = []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Scanning"):
        inputs = inputs.to(DEVICE)

        # 1. Get Logits
        out_32 = model_fp32(inputs)
        out_8 = model_int8(inputs)

        logits_fp32.extend(out_32.numpy())
        logits_int8.extend(out_8.numpy())
        labels_list.extend(labels.numpy())

        # 2. Get Features (Drift Analysis)
        # Note: forward_features returns (Batch, Tokens, Dim) or (Batch, Dim) depending on pooling
        # We average pool manually if needed to get a single vector per image
        f_32 = model_fp32.forward_features(inputs)
        f_8 = model_int8.forward_features(inputs)

        # Swin returns (B, H, W, C) or (B, N, C). We Global Average Pool.
        if f_32.dim() == 4: # (B, H, W, C) -> (B, C)
            f_32 = f_32.mean(dim=(1, 2))
            f_8 = f_8.mean(dim=(1, 2))
        elif f_32.dim() == 3: # (B, N, C) -> (B, C)
            f_32 = f_32.mean(dim=1)
            f_8 = f_8.mean(dim=1)

        features_fp32.extend(f_32.numpy())
        features_int8.extend(f_8.numpy())

# Convert to arrays
probs_fp32 = F.softmax(torch.tensor(np.array(logits_fp32)), dim=1)[:, 1].numpy()
probs_int8 = F.softmax(torch.tensor(np.array(logits_int8)), dim=1)[:, 1].numpy()
y_test = np.array(labels_list)
feats_32 = np.array(features_fp32)
feats_8 = np.array(features_int8)

# --- Step 5: Analysis ---
print("\n" + "="*40)
print("🔬 PHASE 5 RESULTS: THE SILENT FAILURE")
print("="*40)

# A. Feature Drift (Cosine Similarity)
# How much did the internal reasoning change?
# CosSim = (A . B) / (||A|| * ||B||)
norms_32 = np.linalg.norm(feats_32, axis=1)
norms_8 = np.linalg.norm(feats_8, axis=1)
dot_products = np.sum(feats_32 * feats_8, axis=1)
cosine_sims = dot_products / (norms_32 * norms_8 + 1e-10)
avg_drift = 1.0 - np.mean(cosine_sims) # Drift = 1 - Similarity

print(f"🧠 Reasoning Drift (1 - CosSim): {avg_drift:.6f}")
print("   (Higher drift = Model is 'thinking' differently)")

# B. Expected Calibration Error (ECE)
def calc_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i+1])
        if np.sum(bin_mask) > 0:
            bin_acc = np.mean(labels[bin_mask] == (probs[bin_mask] > 0.5))
            bin_conf = np.mean(probs[bin_mask])
            ece += np.abs(bin_conf - bin_acc) * (np.sum(bin_mask) / len(probs))
    return ece

ece_32 = calc_ece(probs_fp32, y_test)
ece_8 = calc_ece(probs_int8, y_test)

print(f"📉 Calibration Error (ECE):")
print(f"   FP32: {ece_32:.4f}")
print(f"   INT8: {ece_8:.4f}")
print(f"   Degradation: {(ece_8 - ece_32)*100:.2f}%")

# --- Step 6: Visualization ---
plt.figure(figsize=(12, 5))

# 1. Reliability Diagram
plt.subplot(1, 2, 1)
prob_true_32, prob_pred_32 = calibration_curve(y_test, probs_fp32, n_bins=10)
prob_true_8, prob_pred_8 = calibration_curve(y_test, probs_int8, n_bins=10)

plt.plot(prob_pred_32, prob_true_32, 'b-o', label=f'FP32 (ECE={ece_32:.3f})')
plt.plot(prob_pred_8, prob_true_8, 'r--s', label=f'INT8 (ECE={ece_8:.3f})')
plt.plot([0, 1], [0, 1], 'k:', label='Perfect Calibration')
plt.title('Reliability Diagram (Calibration)')
plt.xlabel('Mean Predicted Confidence')
plt.ylabel('Fraction of Positives')
plt.legend()
plt.grid(True)

# 2. Drift Histogram
plt.subplot(1, 2, 2)
plt.hist(cosine_sims, bins=50, color='purple', alpha=0.7)
plt.title(f'Internal Reasoning Stability\nAvg Similarity: {np.mean(cosine_sims):.4f}')
plt.xlabel('Cosine Similarity (FP32 vs INT8)')
plt.ylabel('Count')
plt.grid(True)

plt.tight_layout()
plt.savefig('calibration_drift_plot.pdf')
print("\n✅ Analysis Complete. Plot saved as 'calibration_drift_plot.pdf'")

# --- Step 1: Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import timm
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# --- Step 2: Setup (Recap) ---
DEVICE_GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_CPU = torch.device("cpu") # INT8 needs CPU

class SafeMalariaDataset(torch.utils.data.Dataset):
    def __init__(self, parasitized_path, uninfected_path, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        # Limit to 500 images for adversarial attack (computationally heavy)
        p_files = [os.path.join(parasitized_path, f) for f in os.listdir(parasitized_path) if f.lower().endswith('.png')][:250]
        u_files = [os.path.join(uninfected_path, f) for f in os.listdir(uninfected_path) if f.lower().endswith('.png')][:250]
        self.image_paths.extend(p_files)
        self.labels.extend([0] * len(p_files))
        self.image_paths.extend(u_files)
        self.labels.extend([1] * len(u_files))

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform: image = self.transform(image)
            return image, self.labels[idx]
        except: return torch.zeros((3, 224, 224)), self.labels[idx]

# Paths
base_path = '/kaggle/input/cell-images-for-detecting-malaria'
p_dir, u_dir = None, None
for root, dirs, files in os.walk(base_path):
    if os.path.basename(root) == 'Parasitized': p_dir = root
    elif os.path.basename(root) == 'Uninfected': u_dir = root

# Transforms
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset
adv_dataset = SafeMalariaDataset(p_dir, u_dir, transform=test_transform)
adv_loader = DataLoader(adv_dataset, batch_size=1, shuffle=False) # Batch 1 for generating attacks

# --- Step 3: FGSM Attack Function ---
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # We don't clamp here because we normalized.
    # Real-world attacks would clamp to [0,1] before normalization,
    # but for feature-space attacks, this is valid.
    return perturbed_image

# --- Step 4: The Attack Loop ---
print("\n" + "="*40)
print("⚔️ PHASE 6: ADVERSARIAL SECURITY AUDIT")
print("="*40)

# Load Models
print("1️⃣ Loading Models...")
# FP32 (The Victim) - Needs Gradients
model_fp32 = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=2)
model_fp32.load_state_dict(torch.load("/kaggle/working/best_swin_malaria.pth", map_location='cpu'))

# FIX: Create INT8 model BEFORE moving FP32 to GPU to avoid the "input type mismatch" error
# Dynamic quantization requires the model to be on CPU.
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {nn.Linear}, dtype=torch.qint8
)
model_int8.to(DEVICE_CPU)
model_int8.eval()

# NOW move FP32 to GPU for the attack generation
model_fp32.to(DEVICE_GPU)
model_fp32.eval()

# Run Attack
epsilons = [0, 0.01, 0.03, 0.05, 0.1]
accuracies_fp32 = []
accuracies_int8 = []

print("2️⃣ Launching FGSM Attacks...")

for eps in epsilons:
    correct_32 = 0
    correct_8 = 0
    total = 0

    # We only attack samples that the model initially gets right
    # (Attacking a wrong sample is meaningless)

    for data, target in tqdm(adv_loader, desc=f"Epsilon {eps}", leave=False):
        data, target = data.to(DEVICE_GPU), target.to(DEVICE_GPU)
        data.requires_grad = True # Enable grad for input generation

        # 1. Forward pass (FP32)
        output = model_fp32(data)
        init_pred = output.max(1, keepdim=True)[1]

        # If the initial prediction is wrong, dont bother attacking
        if init_pred.item() != target.item():
            continue

        # 2. Calculate Loss & Gradients
        loss = nn.CrossEntropyLoss()(output, target)
        model_fp32.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        # 3. Create Attack Image
        perturbed_data = fgsm_attack(data, eps, data_grad)

        # 4. Test FP32 on Attack (White Box)
        output_32 = model_fp32(perturbed_data)
        final_pred_32 = output_32.max(1, keepdim=True)[1]
        if final_pred_32.item() == target.item():
            correct_32 += 1

        # 5. Test INT8 on Attack (Transfer Attack)
        # Move perturbed data to CPU for INT8
        perturbed_data_cpu = perturbed_data.detach().cpu()
        output_8 = model_int8(perturbed_data_cpu)
        final_pred_8 = output_8.max(1, keepdim=True)[1]
        if final_pred_8.item() == target.item():
            correct_8 += 1

        total += 1

    # Calculate accuracy for this epsilon
    acc_32 = correct_32 / total
    acc_8 = correct_8 / total
    accuracies_fp32.append(acc_32)
    accuracies_int8.append(acc_8)

    print(f"   Epsilon {eps:<4}: FP32 Acc = {acc_32:.4f} | INT8 Acc = {acc_8:.4f} | Diff = {acc_8 - acc_32:.4f}")

# --- Step 5: Visualization ---
plt.figure(figsize=(8, 6))
plt.plot(epsilons, accuracies_fp32, "b-o", label="FP32 (White Box)")
plt.plot(epsilons, accuracies_int8, "r--s", label="INT8 (Transfer)")
plt.title("Adversarial Robustness (FGSM Attack)")
plt.xlabel("Epsilon (Attack Strength)")
plt.ylabel("Accuracy (on initially correct samples)")
plt.legend()
plt.grid(True)
plt.savefig('adversarial_plot.pdf')
print("\n✅ Audit Complete. Plot saved as 'adversarial_plot.pdf'")

# --- Step 1: Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import timm
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import copy

# --- Step 2: Setup (Recap) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU = torch.device("cpu")

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

seed_everything(42)

# Auto-detect paths
base_path = '/kaggle/input/cell-images-for-detecting-malaria'
p_dir, u_dir = None, None
for root, dirs, files in os.walk(base_path):
    if os.path.basename(root) == 'Parasitized': p_dir = root
    elif os.path.basename(root) == 'Uninfected': u_dir = root

# Dataset Class
class SafeMalariaDataset(torch.utils.data.Dataset):
    def __init__(self, parasitized_path, uninfected_path, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        # Use a subset for fast ablation training (3000 images)
        p_files = [os.path.join(parasitized_path, f) for f in os.listdir(parasitized_path) if f.lower().endswith('.png')][:1500]
        u_files = [os.path.join(uninfected_path, f) for f in os.listdir(uninfected_path) if f.lower().endswith('.png')][:1500]
        self.image_paths.extend(p_files)
        self.labels.extend([0] * len(p_files))
        self.image_paths.extend(u_files)
        self.labels.extend([1] * len(u_files))

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform: image = self.transform(image)
            return image, self.labels[idx]
        except: return torch.zeros((3, 224, 224)), self.labels[idx]

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load Data
full_ds = SafeMalariaDataset(p_dir, u_dir, transform=None)
train_size = int(0.8 * len(full_ds))
test_size = len(full_ds) - train_size
train_ds, test_ds = random_split(full_ds, [train_size, test_size])

# Wrappers
class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return self.transform(x), y
    def __len__(self): return len(self.subset)

train_loader = DataLoader(TransformedSubset(train_ds, train_transform), batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(TransformedSubset(test_ds, val_transform), batch_size=1, shuffle=False) # Batch 1 for attacks

# --- Step 3: Train ViT-Tiny (The Ablation Model) ---
print("\n" + "="*40)
print("🔬 PHASE 7: ARCHITECTURE ABLATION (ViT)")
print("="*40)

def train_vit():
    print("1️⃣ Training ViT-Tiny (Comparison Model)...")
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=2)
    model = model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    # Fast training (5 epochs is enough for convergence on this subset)
    for epoch in range(5):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/5", leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"   Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f}")

    return model

vit_fp32 = train_vit()
# Save weights for reproducibility
torch.save(vit_fp32.state_dict(), "vit_tiny_malaria.pth")

# --- Step 4: Quantize ViT ---
print("\n2️⃣ Quantizing ViT (INT8)...")
vit_fp32.cpu() # Move to CPU for quantization
vit_int8 = torch.quantization.quantize_dynamic(
    vit_fp32, {nn.Linear}, dtype=torch.qint8
)

# --- Step 5: Attack Loop (Same as Phase 6) ---
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return perturbed_image

print("\n3️⃣ Running Attack on ViT (Ablation)...")
vit_fp32.to(DEVICE) # Move FP32 back to GPU for gradients
vit_int8.to(CPU)

epsilons = [0, 0.01, 0.03, 0.05, 0.1]
acc_vit_32 = []
acc_vit_8 = []

for eps in epsilons:
    correct_32 = 0
    correct_8 = 0
    total = 0

    # Attack first 200 samples
    for i, (data, target) in enumerate(tqdm(test_loader, desc=f"Eps {eps}", leave=False)):
        if i >= 200: break

        data, target = data.to(DEVICE), target.to(DEVICE)
        data.requires_grad = True

        # FP32 Forward
        output = vit_fp32(data)
        init_pred = output.max(1, keepdim=True)[1]

        if init_pred.item() != target.item(): continue

        loss = nn.CrossEntropyLoss()(output, target)
        vit_fp32.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        # Create Attack
        perturbed_data = fgsm_attack(data, eps, data_grad)

        # Test FP32
        out_32 = vit_fp32(perturbed_data)
        pred_32 = out_32.max(1, keepdim=True)[1]
        if pred_32.item() == target.item(): correct_32 += 1

        # Test INT8
        perturbed_cpu = perturbed_data.detach().cpu()
        out_8 = vit_int8(perturbed_cpu)
        pred_8 = out_8.max(1, keepdim=True)[1]
        if pred_8.item() == target.item(): correct_8 += 1

        total += 1

    acc_32 = correct_32 / total
    acc_8 = correct_8 / total
    acc_vit_32.append(acc_32)
    acc_vit_8.append(acc_8)
    print(f"   Eps {eps}: ViT-FP32={acc_32:.3f} | ViT-INT8={acc_8:.3f}")

# --- Step 6: Combined Visualization (Swin vs ViT) ---
plt.figure(figsize=(10, 6))

# Plot ViT (Ablation)
plt.plot(epsilons, acc_vit_32, "g-o", label="ViT FP32 (White Box)")
plt.plot(epsilons, acc_vit_8, "k--s", label="ViT INT8 (Transfer)")

# Add Swin lines (Reference)
# We plot these faintly to show the comparison
plt.plot(epsilons, [1.0, 0.69, 0.54, 0.52, 0.58], "b-.", alpha=0.5, label="Swin FP32 (Ref)")
plt.plot(epsilons, [1.0, 0.70, 0.54, 0.53, 0.58], "r-.", alpha=0.5, label="Swin INT8 (Ref)")

plt.title("Ablation Study: Is the 'Quantization Shield' Architecture-Agnostic?")
plt.xlabel("Attack Strength (Epsilon)")
plt.ylabel("Robustness (Accuracy)")
plt.legend()
plt.grid(True)
plt.savefig('ablation_study.pdf')
print("\n✅ Ablation Complete. Plot saved as 'ablation_study.pdf'")

# --- Step 1: Imports ---
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# --- Step 2: Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU = torch.device("cpu")

# Dataset (Recap)
class SafeMalariaDataset(torch.utils.data.Dataset):
    def __init__(self, parasitized_path, uninfected_path, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        # Use 500 images for PGD (it is slow)
        p_files = [os.path.join(parasitized_path, f) for f in os.listdir(parasitized_path) if f.lower().endswith('.png')][:250]
        u_files = [os.path.join(uninfected_path, f) for f in os.listdir(uninfected_path) if f.lower().endswith('.png')][:250]
        self.image_paths.extend(p_files)
        self.labels.extend([0] * len(p_files))
        self.image_paths.extend(u_files)
        self.labels.extend([1] * len(u_files))

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform: image = self.transform(image)
            return image, self.labels[idx]
        except: return torch.zeros((3, 224, 224)), self.labels[idx]

# Paths
base_path = '/kaggle/input/cell-images-for-detecting-malaria'
p_dir, u_dir = None, None
for root, dirs, files in os.walk(base_path):
    if os.path.basename(root) == 'Parasitized': p_dir = root
    elif os.path.basename(root) == 'Uninfected': u_dir = root

# Transforms (Standard ImageNet Stats)
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=norm_mean, std=norm_std)
])

# Load Data
audit_ds = SafeMalariaDataset(p_dir, u_dir, transform=test_transform)
audit_loader = DataLoader(audit_ds, batch_size=16, shuffle=False)

# --- Step 3: The PGD Attack (Reviewer's Logic Adapted) ---
def pgd_attack(model, images, labels, eps=0.03, alpha=2/255, steps=20, device='cuda', mean=None, std=None):
    model.eval()

    # Handle Normalization Stats for Inverse/Re-norm
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean).view(1, 3, 1, 1).to(device)
        std_t = torch.tensor(std).view(1, 3, 1, 1).to(device)
        images_raw = images.clone().detach() * std_t + mean_t
        images_raw = torch.clamp(images_raw, 0, 1)
    else:
        mean_t, std_t = None, None
        images_raw = images.clone().detach()

    adv_images_raw = images_raw.clone().detach()
    adv_images_raw.requires_grad = True

    for step in range(steps):
        # Re-normalize
        if mean_t is not None:
            adv_input = (adv_images_raw - mean_t) / std_t
        else:
            adv_input = adv_images_raw

        # Forward
        outputs = model(adv_input)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        # Gradient
        model.zero_grad()
        loss.backward()

        # Update
        grad = adv_images_raw.grad.data
        adv_images_raw.data = adv_images_raw.data + alpha * grad.sign()

        # Project
        eta = torch.clamp(adv_images_raw.data - images_raw, min=-eps, max=eps)
        adv_images_raw.data = torch.clamp(images_raw + eta, min=0, max=1)

        adv_images_raw.grad = None

    # Return Normalized
    if mean_t is not None:
        return (adv_images_raw.detach() - mean_t) / std_t
    else:
        return adv_images_raw.detach()

# --- Step 4: The Safety Audit Loop ---
print("\n" + "="*40)
print("🛡️ PHASE 8: PGD SAFETY AUDIT (Reviewer Request)")
print("="*40)

# Load Swin Models
print("1️⃣ Loading Models...")
model_fp32 = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=2)
model_fp32.load_state_dict(torch.load("/kaggle/working/best_swin_malaria.pth", map_location='cpu'))

# Create INT8 (CPU)
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {nn.Linear}, dtype=torch.qint8
)
model_int8.to(CPU)
model_int8.eval()

# Move FP32 to GPU for Attack Gen
model_fp32.to(DEVICE)
model_fp32.eval()

# Audit
epsilons = [0.01, 0.03, 0.05]
results = {'eps': epsilons, 'fp32': [], 'int8': []}

for eps in epsilons:
    print(f"\nrunning PGD (Eps={eps}, Steps=20)...")
    clean_correct = 0
    adv_correct_fp32 = 0
    adv_correct_int8 = 0
    total = 0

    for images, labels in tqdm(audit_loader, leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        total += labels.size(0)

        # 1. Generate PGD Attack using FP32 Model (White Box)
        # We attack the FP32 model. The question is: Does INT8 resist this attack?
        adv_images = pgd_attack(model_fp32, images, labels, eps=eps, steps=20,
                                device=DEVICE, mean=norm_mean, std=norm_std)

        # 2. Evaluate FP32
        with torch.no_grad():
            out_32 = model_fp32(adv_images)
            pred_32 = out_32.argmax(dim=1)
            adv_correct_fp32 += (pred_32 == labels).sum().item()

        # 3. Evaluate INT8 (Transfer)
        # Move adversarial images to CPU for INT8 inference
        adv_images_cpu = adv_images.cpu()
        with torch.no_grad():
            out_8 = model_int8(adv_images_cpu)
            pred_8 = out_8.argmax(dim=1).to(DEVICE)
            adv_correct_int8 += (pred_8 == labels).sum().item()

    acc_32 = adv_correct_fp32 / total
    acc_8 = adv_correct_int8 / total
    results['fp32'].append(acc_32)
    results['int8'].append(acc_8)

    print(f"   [Eps {eps}] FP32 Acc: {acc_32:.4f} | INT8 Acc: {acc_8:.4f} | Delta: {acc_8 - acc_32:.4f}")

# --- Step 5: Visualization ---
plt.figure(figsize=(8, 6))
plt.plot(results['eps'], results['fp32'], 'b-o', label='FP32 (Victim)')
plt.plot(results['eps'], results['int8'], 'r--s', label='INT8 (Transfer Defense)')
plt.title("Robustness against Iterative PGD Attack (20 Steps)")
plt.xlabel("Attack Strength (Epsilon)")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig('pgd_audit_plot.pdf')
print("\n✅ PGD Audit Complete. Check 'pgd_audit_plot.pdf'.")

# --- Step 1: Train MobileNetV3 ---
print("\n" + "="*40)
print("📱 PHASE 9: MOBILENETV3 BASELINE")
print("="*40)

def train_mobilenet():
    # 1. Load Data (Same as Phase 1)
    # Note: Ensure get_dataloaders is defined or re-copy from Phase 1 if session restarted
    # For brevity, we assume 'train_loader' etc. are available or we re-init
    # Re-init loaders just in case:
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, random_split
    import os

    base_path = '/kaggle/input/cell-images-for-detecting-malaria'
    # Auto-find (Assuming code from Phase 1 ran)
    p_dir, u_dir = None, None
    for root, dirs, files in os.walk(base_path):
        if 'Parasitized' in dirs: p_dir = os.path.join(root, 'Parasitized')
        if 'Uninfected' in dirs: u_dir = os.path.join(root, 'Uninfected')

    # Simple loader for speed
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # We need a quick Train Loop just for MobileNet
    # We will use the 'SafeMalariaDataset' class defined in Phase 4/5/6
    # If not defined in current scope, define it:
    class QuickDataset(torch.utils.data.Dataset):
        def __init__(self, p_path, u_path, trans):
            self.files = ([os.path.join(p_path, f) for f in os.listdir(p_path) if f.lower().endswith('.png')][:2000] +
                          [os.path.join(u_path, f) for f in os.listdir(u_path) if f.lower().endswith('.png')][:2000])
            self.labels = [0]*2000 + [1]*2000
            self.trans = trans
        def __len__(self): return len(self.files)
        def __getitem__(self, i):
            try: return self.trans(Image.open(self.files[i]).convert('RGB')), self.labels[i]
            except: return torch.zeros(3,224,224), self.labels[i]

    ds = QuickDataset(p_dir, u_dir, transform)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    # 2. Model
    print("Loading MobileNetV3-Small...")
    model = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=2)
    model = model.to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    # Train 5 epochs
    for ep in range(5):
        model.train()
        run_loss = 0
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            run_loss += loss.item()
        print(f"   Epoch {ep+1} Loss: {run_loss/len(loader):.4f}")

    return model

# Train
mobilenet = train_mobilenet()

# --- Step 2: Compare Size & Quantization ---
mobilenet.cpu()
torch.save(mobilenet.state_dict(), "mobilenet_temp.pth")
size_mb = os.path.getsize("mobilenet_temp.pth") / 1e6
print(f"MobileNet FP32 Size: {size_mb:.2f} MB")

# Quantize
mobilenet_int8 = torch.quantization.quantize_dynamic(
    mobilenet, {nn.Linear}, dtype=torch.qint8
)
torch.save(mobilenet_int8.state_dict(), "mobilenet_int8_temp.pth")
size_int8 = os.path.getsize("mobilenet_int8_temp.pth") / 1e6
print(f"MobileNet INT8 Size: {size_int8:.2f} MB")

# Compare with Swin (Hardcoded from Phase 2 for context)
print("-" * 30)
print(f"Swin-Tiny INT8 Size: ~28 MB")
print(f"MobileNet INT8 Size: {size_int8:.2f} MB")
print("Note: MobileNet is smaller, BUT does it have the 'Shield'?")

# --- Step 3: Quick PGD Check on MobileNet ---
# Does MobileNet resist PGD like Swin?
mobilenet.to(DEVICE)
mobilenet.eval()
mobilenet_int8.to(CPU)
mobilenet_int8.eval()

# Use the PGD function from Phase 8
# (Assuming variables/functions from Phase 8 are still in memory)
print("\nRunning PGD on MobileNet (Eps=0.03)...")
correct_32 = 0
correct_8 = 0
total = 0

# Use a small batch for speed
for images, labels in audit_loader:
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    if total > 200: break # Quick check
    total += labels.size(0)

    adv = pgd_attack(mobilenet, images, labels, eps=0.03, steps=20, device=DEVICE, mean=norm_mean, std=norm_std)

    # FP32
    with torch.no_grad():
        pred_32 = mobilenet(adv).argmax(1)
        correct_32 += (pred_32 == labels).sum().item()

    # INT8
    with torch.no_grad():
        pred_8 = mobilenet_int8(adv.cpu()).argmax(1).to(DEVICE)
        correct_8 += (pred_8 == labels).sum().item()

print(f"MobileNet FP32 PGD Acc: {correct_32/total:.2f}")
print(f"MobileNet INT8 PGD Acc: {correct_8/total:.2f}")
print("Conclusion: Check if MobileNet INT8 provides the same 'Shield' benefit.")

# --- Step 1: Imports ---
import torch
import torch.nn as nn
import numpy as np
import timm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm.notebook import tqdm

# --- Step 2: Setup OOD Dataset (BCCD White Blood Cells) ---
print("📥 Preparing OOD Dataset (White Blood Cells)...")

# Standard ImageNet Norm (Matches the Malaria Training)
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Setup dataset path with fallback
wbc_dir = '/kaggle/input/bccd-white-blood-cell/bccd_wbc'

# Check if the specific path exists, if not, try auto-discovery
if not os.path.exists(wbc_dir):
    print(f"⚠️ Specific path {wbc_dir} not found. Attempting auto-discovery...")
    base_kaggle_path = '/kaggle/input'
    wbc_dir = None

    for root, dirs, files in os.walk(base_kaggle_path):
        # We look for the folder containing the specific WBC classes
        if 'neutrophil' in dirs and 'monocyte' in dirs:
            wbc_dir = root
            break

    if wbc_dir is None:
        # Fallback search if folder names are capitalized or different
        for root, dirs, files in os.walk(base_kaggle_path):
            if 'Neutrophil' in dirs or 'NEUTROPHIL' in dirs:
                wbc_dir = root
                break

if wbc_dir and os.path.exists(wbc_dir):
    print(f"✅ Found WBC Data at: {wbc_dir}")
else:
    raise FileNotFoundError("Could not find BCCD/WBC folders. Check dataset mounting.")

# Load WBC as "Unknowns"
ood_dataset = datasets.ImageFolder(root=wbc_dir, transform=eval_transform)
ood_loader = DataLoader(ood_dataset, batch_size=32, shuffle=False, num_workers=2)

print(f"📊 OOD Stats: {len(ood_dataset)} White Blood Cells (Basophil, Eosinophil, etc.)")

# --- Step 3: Energy Score Function ---
def get_energy_score(logits, T=1.0):
    # Energy = -T * log(sum(exp(logits/T)))
    # Lower energy = In-Distribution (Malaria)
    # Higher energy = OOD (WBC)
    return -T * torch.logsumexp(logits / T, dim=1)

def extract_scores(model, loader, device="cuda"):
    model.to(device)
    model.eval()
    scores = []

    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc="Scoring", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Negative Energy = Confidence
            energy = get_energy_score(outputs)
            scores.extend((-energy).cpu().numpy())

    return np.array(scores)

# --- Step 4: The Audit ---
print("\n" + "="*40)
print("🕵️ PHASE 3 (REV): CLINICAL RELIABILITY AUDIT")
print("Comparing Malaria (ID) vs. White Blood Cells (OOD)")
print("="*40)

# Setup Malaria Loader (ID) - Re-using code from Phase 2/5 for loader creation
# Quick re-def to ensure this cell runs standalone
class SafeMalariaDataset(torch.utils.data.Dataset):
    def __init__(self, p_path, u_path, transform):
        self.files = ([os.path.join(p_path, f) for f in os.listdir(p_path) if f.lower().endswith('.png')] +
                      [os.path.join(u_path, f) for f in os.listdir(u_path) if f.lower().endswith('.png')])
        self.labels = [0]*len([f for f in os.listdir(p_path) if f.endswith('.png')]) + [1]*len([f for f in os.listdir(u_path) if f.endswith('.png')])
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        from PIL import Image
        try: return self.transform(Image.open(self.files[i]).convert('RGB')), self.labels[i]
        except: return torch.zeros(3,224,224), self.labels[i]

# Find Malaria Data
malaria_root = '/kaggle/input/cell-images-for-detecting-malaria'
p_dir, u_dir = None, None
for root, dirs, files in os.walk(malaria_root):
    if 'Parasitized' in dirs: p_dir = os.path.join(root, 'Parasitized')
    if 'Uninfected' in dirs: u_dir = os.path.join(root, 'Uninfected')

# Use a subset of Test data for ID to match OOD size roughly
id_dataset = SafeMalariaDataset(p_dir, u_dir, eval_transform)
# Just take last 20% as pseudo-test if not explicitly split
id_subset_len = int(len(id_dataset) * 0.2)
from torch.utils.data import random_split
_, id_test_ds = random_split(id_dataset, [len(id_dataset)-id_subset_len, id_subset_len], generator=torch.Generator().manual_seed(42))
id_loader = DataLoader(id_test_ds, batch_size=32, shuffle=False, num_workers=2)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Models
print("1️⃣ Loading Models...")
model_fp32 = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=2)
model_fp32.load_state_dict(torch.load("/kaggle/working/best_swin_malaria.pth", map_location='cpu'))

# Dynamic INT8
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {nn.Linear}, dtype=torch.qint8
)

# 1. Score Malaria
print("2️⃣ Scoring In-Distribution (Malaria)...")
id_scores_fp32 = extract_scores(model_fp32, id_loader, device=DEVICE)
id_scores_int8 = extract_scores(model_int8, id_loader, device="cpu") # INT8 on CPU

# 2. Score WBC
print("3️⃣ Scoring Out-of-Distribution (WBCs)...")
ood_scores_fp32 = extract_scores(model_fp32, ood_loader, device=DEVICE)
ood_scores_int8 = extract_scores(model_int8, ood_loader, device="cpu")

# --- Step 5: Results ---
def calc_ood_auroc(id_scores, ood_scores):
    y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    y_scores = np.concatenate([id_scores, ood_scores])
    return roc_auc_score(y_true, y_scores)

auroc_fp32 = calc_ood_auroc(id_scores_fp32, ood_scores_fp32)
auroc_int8 = calc_ood_auroc(id_scores_int8, ood_scores_int8)

print("\n" + "="*40)
print("📉 CLINICAL RELIABILITY RESULTS (AUROC)")
print("Ability to distinguish Malaria vs. White Blood Cells")
print("="*40)
print(f"FP32 AUROC:       {auroc_fp32:.4f}")
print(f"INT8 AUROC:       {auroc_int8:.4f}")
print(f"Reliability Drop: {(auroc_fp32 - auroc_int8):.4f}")
print("-" * 40)

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.kdeplot(id_scores_fp32, fill=True, label='Malaria', color='blue')
sns.kdeplot(ood_scores_fp32, fill=True, label='WBC (Artifact)', color='red')
plt.title(f'FP32 Reliability\nAUROC: {auroc_fp32:.3f}')
plt.legend()

plt.subplot(1, 2, 2)
sns.kdeplot(id_scores_int8, fill=True, label='Malaria', color='blue')
sns.kdeplot(ood_scores_int8, fill=True, label='WBC (Artifact)', color='red')
plt.title(f'INT8 Reliability\nAUROC: {auroc_int8:.3f}')
plt.legend()

plt.tight_layout()
plt.savefig('clinical_reliability_wbc.pdf')
print("✅ Plot saved as 'clinical_reliability_wbc.pdf'")

# --- Step 1: Imports ---
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
import os
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Step 2: Define Corruptions (Same as Phase 4) ---
class AddGaussianNoise(object):
    def __init__(self, severity=1):
        self.std = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10][severity]
    def __call__(self, tensor):
        if self.std == 0: return tensor
        return tensor + torch.randn(tensor.size()) * self.std

class AddDefocusBlur(object):
    def __init__(self, severity=1):
        self.radius = [0, 0.5, 1.0, 1.5, 2.0, 2.5][severity]
    def __call__(self, img):
        if self.radius == 0: return img
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))

# --- Step 3: Dataset & Loader ---
class SafeMalariaDataset(torch.utils.data.Dataset):
    def __init__(self, p_path, u_path, transform):
        # Use 1000 images for robust stats
        self.files = ([os.path.join(p_path, f) for f in os.listdir(p_path) if f.lower().endswith('.png')][:500] +
                      [os.path.join(u_path, f) for f in os.listdir(u_path) if f.lower().endswith('.png')][:500])
        self.labels = [0]*500 + [1]*500
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        try: return self.transform(Image.open(self.files[i]).convert('RGB')), self.labels[i]
        except: return torch.zeros(3,224,224), self.labels[i]

# Auto-detect paths
base_path = '/kaggle/input/cell-images-for-detecting-malaria'
p_dir, u_dir = None, None
for root, dirs, files in os.walk(base_path):
    if 'Parasitized' in dirs: p_dir = os.path.join(root, 'Parasitized')
    if 'Uninfected' in dirs: u_dir = os.path.join(root, 'Uninfected')

def get_dataloader(severity, corruption_type):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    if corruption_type == 'noise':
        t = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            AddGaussianNoise(severity),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])
    elif corruption_type == 'blur':
        t = transforms.Compose([
            transforms.Resize((224, 224)),
            AddDefocusBlur(severity),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])

    ds = SafeMalariaDataset(p_dir, u_dir, transform=t)
    return DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

# --- Step 4: The Comparison Loop ---
print("\n" + "="*40)
print("🏥 PHASE 10: CLINICAL ROBUSTNESS (MobileNet vs Swin)")
print("="*40)

# Load Models
print("1️⃣ Loading Models...")
# MobileNet (Train from scratch or load if saved)
# Note: Re-instantiating and quick-training if weight file missing,
# For this script, we assume 'mobilenet_temp.pth' exists from Phase 9.
# If not, we quickly re-train a proxy or load pretrained.
try:
    mobilenet = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=2)
    mobilenet.load_state_dict(torch.load("mobilenet_temp.pth"))
except:
    print("Warning: Trained MobileNet weights not found. Loading Pretrained (ImageNet) for demo.")
    mobilenet = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=2)

mobilenet.to(DEVICE)
mobilenet.eval()

# Swin (FP32)
swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=2)
swin.load_state_dict(torch.load("/kaggle/working/best_swin_malaria.pth", map_location='cpu'))
swin.to(DEVICE)
swin.eval()

# Run Tests
severities = [0, 1, 2, 3, 4, 5]
results = {'blur_mob': [], 'blur_swin': [], 'noise_mob': [], 'noise_swin': []}

def eval_model(model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total

print("\n2️⃣ Testing Defocus Blur...")
for s in severities:
    loader = get_dataloader(s, 'blur')
    acc_mob = eval_model(mobilenet, loader)
    acc_swin = eval_model(swin, loader)
    results['blur_mob'].append(acc_mob)
    results['blur_swin'].append(acc_swin)
    print(f"   Sev {s}: MobileNet={acc_mob:.3f} | Swin={acc_swin:.3f}")

print("\n3️⃣ Testing Sensor Noise...")
for s in severities:
    loader = get_dataloader(s, 'noise')
    acc_mob = eval_model(mobilenet, loader)
    acc_swin = eval_model(swin, loader)
    results['noise_mob'].append(acc_mob)
    results['noise_swin'].append(acc_swin)
    print(f"   Sev {s}: MobileNet={acc_mob:.3f} | Swin={acc_swin:.3f}")

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(severities, results['blur_mob'], 'r--o', label='MobileNet')
plt.plot(severities, results['blur_swin'], 'b-s', label='Swin Transformer')
plt.title("Robustness to Defocus Blur")
plt.xlabel("Severity")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(severities, results['noise_mob'], 'r--o', label='MobileNet')
plt.plot(severities, results['noise_swin'], 'b-s', label='Swin Transformer')
plt.title("Robustness to Sensor Noise")
plt.xlabel("Severity")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('clinical_comparison.pdf')
print("\n✅ Comparison Complete. Plot saved as 'clinical_comparison.pdf'")

# --- Step 1: Imports ---
import torch
import torch.nn as nn
import numpy as np
import timm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm.notebook import tqdm

# --- Step 2: Setup OOD Dataset (BCCD White Blood Cells) ---
print("📥 Preparing OOD Dataset (White Blood Cells)...")

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Specific path provided by user
wbc_dir = '/kaggle/input/bccd-white-blood-cell/bccd_wbc'

# Fallback discovery if specific path fails
if not os.path.exists(wbc_dir):
    print(f"⚠️ Path {wbc_dir} not found. Attempting auto-discovery...")
    base_kaggle_path = '/kaggle/input'
    wbc_dir = None
    for root, dirs, files in os.walk(base_kaggle_path):
        if 'neutrophil' in dirs and 'monocyte' in dirs:
            wbc_dir = root
            break
    if wbc_dir is None:
        for root, dirs, files in os.walk(base_kaggle_path):
            if 'Neutrophil' in dirs or 'NEUTROPHIL' in dirs:
                wbc_dir = root
                break

if wbc_dir and os.path.exists(wbc_dir):
    print(f"✅ Found WBC Data at: {wbc_dir}")
else:
    raise FileNotFoundError("Could not find BCCD/WBC folders. Check dataset mounting.")

ood_dataset = datasets.ImageFolder(root=wbc_dir, transform=eval_transform)
ood_loader = DataLoader(ood_dataset, batch_size=32, shuffle=False, num_workers=2)

print(f"📊 OOD Stats: {len(ood_dataset)} White Blood Cells (Basophil, Eosinophil, etc.)")

# --- Step 3: Helper Functions ---
def get_energy_score(logits, T=1.0):
    # Energy = -T * log(sum(exp(logits/T)))
    # We use negative energy as the "confidence" score.
    # Higher Score = More likely to be Malaria (ID).
    # Lower Score = More likely to be WBC (OOD).
    return -T * torch.logsumexp(logits / T, dim=1)

def extract_scores(model, loader, device="cuda"):
    model.to(device)
    model.eval()
    scores = []
    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc="Scoring", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            energy = get_energy_score(outputs)
            scores.extend((-energy).cpu().numpy())
    return np.array(scores)

def calculate_fpr95(y_true, y_scores):
    """
    Calculates False Positive Rate at 95% True Positive Rate (Recall).
    y_true: 1 for ID (Malaria), 0 for OOD (WBC)
    y_scores: Confidence scores (higher is ID)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # Find the index where TPR is closest to 0.95 (95% sensitivity)
    target_tpr = 0.95
    idx = np.argmin(np.abs(tpr - target_tpr))
    return fpr[idx], thresholds[idx]

def run_mcnemars_test(model_1_preds, model_2_preds, true_labels):
    """
    Compares two models using McNemar's Test.
    Checks if the disagreement between models is statistically significant.
    """
    # Contingency Table:
    # [Both Correct, M1 Correct & M2 Wrong]
    # [M1 Wrong & M2 Correct, Both Wrong]

    both_correct = sum((p1 == t and p2 == t) for p1, p2, t in zip(model_1_preds, model_2_preds, true_labels))
    only_m1_correct = sum((p1 == t and p2 != t) for p1, p2, t in zip(model_1_preds, model_2_preds, true_labels))
    only_m2_correct = sum((p1 != t and p2 == t) for p1, p2, t in zip(model_1_preds, model_2_preds, true_labels))
    both_wrong = sum((p1 != t and p2 != t) for p1, p2, t in zip(model_1_preds, model_2_preds, true_labels))

    table = [[both_correct, only_m1_correct],
             [only_m2_correct, both_wrong]]

    # Exact test is better for smaller contingency counts
    result = mcnemar(table, exact=True)
    return result.pvalue

# --- Step 4: The Audit ---
print("\n" + "="*40)
print("🕵️ PHASE 3 (REV): STATISTICAL CLINICAL AUDIT")
print("Comparing Malaria (ID) vs. White Blood Cells (OOD)")
print("="*40)

# Dataset Setup (Malaria ID)
class SafeMalariaDataset(torch.utils.data.Dataset):
    def __init__(self, p_path, u_path, transform):
        # We manually list files to avoid the duplicate folder issue from Phase 1
        self.files = ([os.path.join(p_path, f) for f in os.listdir(p_path) if f.lower().endswith('.png')] +
                      [os.path.join(u_path, f) for f in os.listdir(u_path) if f.lower().endswith('.png')])
        # 0 = Parasitized, 1 = Uninfected (Though for OOD detection, both are "ID")
        self.labels = [0]*len([f for f in os.listdir(p_path) if f.endswith('.png')]) + [1]*len([f for f in os.listdir(u_path) if f.endswith('.png')])
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        from PIL import Image
        try: return self.transform(Image.open(self.files[i]).convert('RGB')), self.labels[i]
        except: return torch.zeros(3,224,224), self.labels[i]

# Find Malaria Data Paths
malaria_root = '/kaggle/input/cell-images-for-detecting-malaria'
p_dir, u_dir = None, None
for root, dirs, files in os.walk(malaria_root):
    if 'Parasitized' in dirs: p_dir = os.path.join(root, 'Parasitized')
    if 'Uninfected' in dirs: u_dir = os.path.join(root, 'Uninfected')

if not p_dir or not u_dir:
    raise FileNotFoundError("Malaria dataset not found.")

# Use 20% subset of Malaria data as "Test ID" for speed and balance
id_dataset = SafeMalariaDataset(p_dir, u_dir, eval_transform)
id_subset_len = int(len(id_dataset) * 0.2)
from torch.utils.data import random_split
_, id_test_ds = random_split(id_dataset, [len(id_dataset)-id_subset_len, id_subset_len], generator=torch.Generator().manual_seed(42))
id_loader = DataLoader(id_test_ds, batch_size=32, shuffle=False, num_workers=2)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Models
print("1️⃣ Loading Models...")
# Load Baseline FP32
model_fp32 = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=2)
model_fp32.load_state_dict(torch.load("/kaggle/working/best_swin_malaria.pth", map_location='cpu'))

# Create INT8 Model (Dynamic Quantization)
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {nn.Linear}, dtype=torch.qint8
)

# Score Extraction
print("2️⃣ Scoring In-Distribution (Malaria)...")
id_scores_fp32 = extract_scores(model_fp32, id_loader, device=DEVICE)
# INT8 must run on CPU
id_scores_int8 = extract_scores(model_int8, id_loader, device="cpu")

print("3️⃣ Scoring Out-of-Distribution (WBCs)...")
ood_scores_fp32 = extract_scores(model_fp32, ood_loader, device=DEVICE)
ood_scores_int8 = extract_scores(model_int8, ood_loader, device="cpu")

# --- Step 5: Statistical Analysis ---
print("\n" + "="*40)
print("📊 STATISTICAL RESULTS")
print("="*40)

# Prepare Labels for ROC/FPR calculation
# 1 = ID (Malaria), 0 = OOD (WBC)
y_true_id = np.ones(len(id_scores_fp32))
y_true_ood = np.zeros(len(ood_scores_fp32))
y_true = np.concatenate([y_true_id, y_true_ood])

y_scores_fp32 = np.concatenate([id_scores_fp32, ood_scores_fp32])
y_scores_int8 = np.concatenate([id_scores_int8, ood_scores_int8])

# 1. FPR95 Calculation
fpr95_fp32, thresh_fp32 = calculate_fpr95(y_true, y_scores_fp32)
fpr95_int8, thresh_int8 = calculate_fpr95(y_true, y_scores_int8)

print(f"FP32 FPR95: {fpr95_fp32:.4f} (at threshold {thresh_fp32:.2f})")
print(f"INT8 FPR95: {fpr95_int8:.4f} (at threshold {thresh_int8:.2f})")
print(f"Delta:      {fpr95_int8 - fpr95_fp32:.4f}")

# 2. McNemar's Test
# We generate binary predictions based on the 95% sensitivity threshold calculated above.
# If Score > Threshold -> Predict ID (1). Else -> Predict OOD (0).
# This creates a "fair" comparison where both models are tuned to the same Recall.
preds_fp32 = (y_scores_fp32 >= thresh_fp32).astype(int)
preds_int8 = (y_scores_int8 >= thresh_int8).astype(int)

p_value = run_mcnemars_test(preds_fp32, preds_int8, y_true)

print("-" * 30)
print(f"McNemar's P-Value: {p_value:.5f}")
if p_value < 0.05:
    print(">> RESULT: The performance difference is STATISTICALLY SIGNIFICANT.")
else:
    print(">> RESULT: The performance difference is NOT significant (Models are statistically equivalent).")

# 3. AUROC (Standard)
auroc_fp32 = roc_auc_score(y_true, y_scores_fp32)
auroc_int8 = roc_auc_score(y_true, y_scores_int8)
print("-" * 30)
print(f"FP32 AUROC: {auroc_fp32:.4f}")
print(f"INT8 AUROC: {auroc_int8:.4f}")

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.kdeplot(id_scores_fp32, fill=True, label='Malaria', color='blue')
sns.kdeplot(ood_scores_fp32, fill=True, label='WBC (Artifact)', color='red')
plt.axvline(thresh_fp32, color='k', linestyle='--', label='95% TPR Thresh')
plt.title(f'FP32 Reliability (FPR95={fpr95_fp32:.3f})')
plt.legend()

plt.subplot(1, 2, 2)
sns.kdeplot(id_scores_int8, fill=True, label='Malaria', color='blue')
sns.kdeplot(ood_scores_int8, fill=True, label='WBC (Artifact)', color='red')
plt.axvline(thresh_int8, color='k', linestyle='--', label='95% TPR Thresh')
plt.title(f'INT8 Reliability (FPR95={fpr95_int8:.3f})')
plt.legend()

plt.tight_layout()
plt.savefig('clinical_reliability_wbc_stats.pdf')
print("✅ Plot saved as 'clinical_reliability_wbc_stats.pdf'")

# --- Step 1: Imports ---
import torch
import torch.nn as nn
import numpy as np
import timm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from tqdm.notebook import tqdm

# --- Step 2: Setup OOD Dataset (BCCD White Blood Cells) ---
print("📥 Preparing OOD Dataset (White Blood Cells)...")

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Specific path provided by user
wbc_dir = '/kaggle/input/bccd-white-blood-cell/bccd_wbc'

# Fallback discovery if specific path fails
if not os.path.exists(wbc_dir):
    print(f"⚠️ Path {wbc_dir} not found. Attempting auto-discovery...")
    base_kaggle_path = '/kaggle/input'
    wbc_dir = None
    for root, dirs, files in os.walk(base_kaggle_path):
        if 'neutrophil' in dirs and 'monocyte' in dirs:
            wbc_dir = root
            break
    if wbc_dir is None:
        for root, dirs, files in os.walk(base_kaggle_path):
            if 'Neutrophil' in dirs or 'NEUTROPHIL' in dirs:
                wbc_dir = root
                break

if wbc_dir and os.path.exists(wbc_dir):
    print(f"✅ Found WBC Data at: {wbc_dir}")
else:
    raise FileNotFoundError("Could not find BCCD/WBC folders. Check dataset mounting.")

ood_dataset = datasets.ImageFolder(root=wbc_dir, transform=eval_transform)
ood_loader = DataLoader(ood_dataset, batch_size=32, shuffle=False, num_workers=2)

print(f"📊 OOD Stats: {len(ood_dataset)} White Blood Cells (Basophil, Eosinophil, etc.)")

# --- Step 3: Helper Functions ---
def get_energy_score(logits, T=1.0):
    # Energy = -T * log(sum(exp(logits/T)))
    # We use negative energy as the "confidence" score.
    # Higher Score = More likely to be Malaria (ID).
    # Lower Score = More likely to be WBC (OOD).
    return -T * torch.logsumexp(logits / T, dim=1)

def extract_scores(model, loader, device="cuda"):
    model.to(device)
    model.eval()
    scores = []
    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc="Scoring", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            energy = get_energy_score(outputs)
            scores.extend((-energy).cpu().numpy())
    return np.array(scores)

def calculate_fpr95(y_true, y_scores):
    """
    Calculates False Positive Rate at 95% True Positive Rate (Recall).
    y_true: 1 for ID (Malaria), 0 for OOD (WBC)
    y_scores: Confidence scores (higher is ID)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # Find the index where TPR is closest to 0.95 (95% sensitivity)
    target_tpr = 0.95
    idx = np.argmin(np.abs(tpr - target_tpr))
    return fpr[idx], thresholds[idx]

def run_mcnemars_test(model_1_preds, model_2_preds, true_labels):
    """
    Compares two models using McNemar's Test.
    Checks if the disagreement between models is statistically significant.
    """
    # Contingency Table:
    # [Both Correct, M1 Correct & M2 Wrong]
    # [M1 Wrong & M2 Correct, Both Wrong]

    both_correct = sum((p1 == t and p2 == t) for p1, p2, t in zip(model_1_preds, model_2_preds, true_labels))
    only_m1_correct = sum((p1 == t and p2 != t) for p1, p2, t in zip(model_1_preds, model_2_preds, true_labels))
    only_m2_correct = sum((p1 != t and p2 == t) for p1, p2, t in zip(model_1_preds, model_2_preds, true_labels))
    both_wrong = sum((p1 != t and p2 != t) for p1, p2, t in zip(model_1_preds, model_2_preds, true_labels))

    table = [[both_correct, only_m1_correct],
             [only_m2_correct, both_wrong]]

    # Exact test is better for smaller contingency counts
    result = mcnemar(table, exact=True)
    return result.pvalue

# --- Step 4: The Audit ---
print("\n" + "="*40)
print("🕵️ PHASE 3 (REV): STATISTICAL CLINICAL AUDIT")
print("Comparing Malaria (ID) vs. White Blood Cells (OOD)")
print("="*40)

# Dataset Setup (Malaria ID)
class SafeMalariaDataset(torch.utils.data.Dataset):
    def __init__(self, p_path, u_path, transform):
        # We manually list files to avoid the duplicate folder issue from Phase 1
        self.files = ([os.path.join(p_path, f) for f in os.listdir(p_path) if f.lower().endswith('.png')] +
                      [os.path.join(u_path, f) for f in os.listdir(u_path) if f.lower().endswith('.png')])
        # 0 = Parasitized, 1 = Uninfected (Though for OOD detection, both are "ID")
        self.labels = [0]*len([f for f in os.listdir(p_path) if f.endswith('.png')]) + [1]*len([f for f in os.listdir(u_path) if f.endswith('.png')])
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        from PIL import Image
        try: return self.transform(Image.open(self.files[i]).convert('RGB')), self.labels[i]
        except: return torch.zeros(3,224,224), self.labels[i]

# Find Malaria Data Paths
malaria_root = '/kaggle/input/cell-images-for-detecting-malaria'
p_dir, u_dir = None, None
for root, dirs, files in os.walk(malaria_root):
    if 'Parasitized' in dirs: p_dir = os.path.join(root, 'Parasitized')
    if 'Uninfected' in dirs: u_dir = os.path.join(root, 'Uninfected')

if not p_dir or not u_dir:
    raise FileNotFoundError("Malaria dataset not found.")

# Use 20% subset of Malaria data as "Test ID" for speed and balance
id_dataset = SafeMalariaDataset(p_dir, u_dir, eval_transform)
id_subset_len = int(len(id_dataset) * 0.2)
from torch.utils.data import random_split
_, id_test_ds = random_split(id_dataset, [len(id_dataset)-id_subset_len, id_subset_len], generator=torch.Generator().manual_seed(42))
id_loader = DataLoader(id_test_ds, batch_size=32, shuffle=False, num_workers=2)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Models
print("1️⃣ Loading Models...")
# Load Baseline FP32
model_fp32 = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=2)
model_fp32.load_state_dict(torch.load("/kaggle/working/best_swin_malaria.pth", map_location='cpu'))

# Create INT8 Model (Dynamic Quantization)
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {nn.Linear}, dtype=torch.qint8
)

# Load MobileNet Baseline for Latency Comparison
print("   Loading MobileNet Baseline for Latency Benchmark...")
mobilenet_fp32 = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=2)
# Weights do not matter for inference time, so we just initialize the architecture
mobilenet_int8 = torch.quantization.quantize_dynamic(
    mobilenet_fp32, {nn.Linear}, dtype=torch.qint8
)

# --- NEW: CPU Latency Benchmark ---
print("\n⏱️ Running CPU Latency Benchmark (Mobile vs. Swin)...")
def measure_cpu_latency(model, loader, num_samples=100):
    model.to("cpu")
    model.eval()

    # Warmup pass
    with torch.no_grad():
        for inputs, _ in loader:
            _ = model(inputs.to("cpu"))
            break

    # Benchmark pass
    start_time = time.time()
    count = 0
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to("cpu")
            if count + inputs.size(0) > num_samples:
                inputs = inputs[:num_samples - count]

            _ = model(inputs)
            count += inputs.size(0)

            if count >= num_samples:
                break

    return ((time.time() - start_time) / count) * 1000

latency_swin_fp32 = measure_cpu_latency(model_fp32, id_loader)
latency_swin_int8 = measure_cpu_latency(model_int8, id_loader)
latency_mob_fp32 = measure_cpu_latency(mobilenet_fp32, id_loader)
latency_mob_int8 = measure_cpu_latency(mobilenet_int8, id_loader)

print(f"   Swin FP32 CPU Latency:       {latency_swin_fp32:.2f} ms / image")
print(f"   Swin INT8 CPU Latency:       {latency_swin_int8:.2f} ms / image")
print(f"   MobileNet FP32 CPU Latency:  {latency_mob_fp32:.2f} ms / image")
print(f"   MobileNet INT8 CPU Latency:  {latency_mob_int8:.2f} ms / image")
print("-" * 40)

# Score Extraction
print("\n2️⃣ Scoring In-Distribution (Malaria)...")
id_scores_fp32 = extract_scores(model_fp32, id_loader, device=DEVICE)
# INT8 must run on CPU
id_scores_int8 = extract_scores(model_int8, id_loader, device="cpu")

print("3️⃣ Scoring Out-of-Distribution (WBCs)...")
ood_scores_fp32 = extract_scores(model_fp32, ood_loader, device=DEVICE)
ood_scores_int8 = extract_scores(model_int8, ood_loader, device="cpu")

# --- Step 5: Statistical Analysis ---
print("\n" + "="*40)
print("📊 STATISTICAL RESULTS")
print("="*40)

# Prepare Labels for ROC/FPR calculation
# 1 = ID (Malaria), 0 = OOD (WBC)
y_true_id = np.ones(len(id_scores_fp32))
y_true_ood = np.zeros(len(ood_scores_fp32))
y_true = np.concatenate([y_true_id, y_true_ood])

y_scores_fp32 = np.concatenate([id_scores_fp32, ood_scores_fp32])
y_scores_int8 = np.concatenate([id_scores_int8, ood_scores_int8])

# 1. FPR95 Calculation
fpr95_fp32, thresh_fp32 = calculate_fpr95(y_true, y_scores_fp32)
fpr95_int8, thresh_int8 = calculate_fpr95(y_true, y_scores_int8)

print(f"FP32 FPR95: {fpr95_fp32:.4f} (at threshold {thresh_fp32:.2f})")
print(f"INT8 FPR95: {fpr95_int8:.4f} (at threshold {thresh_int8:.2f})")
print(f"Delta:      {fpr95_int8 - fpr95_fp32:.4f}")

# 2. McNemar's Test
# We generate binary predictions based on the 95% sensitivity threshold calculated above.
# If Score > Threshold -> Predict ID (1). Else -> Predict OOD (0).
# This creates a "fair" comparison where both models are tuned to the same Recall.
preds_fp32 = (y_scores_fp32 >= thresh_fp32).astype(int)
preds_int8 = (y_scores_int8 >= thresh_int8).astype(int)

p_value = run_mcnemars_test(preds_fp32, preds_int8, y_true)

print("-" * 30)
print(f"McNemar's P-Value: {p_value:.5f}")
if p_value < 0.05:
    print(">> RESULT: The performance difference is STATISTICALLY SIGNIFICANT.")
else:
    print(">> RESULT: The performance difference is NOT significant (Models are statistically equivalent).")

# 3. AUROC (Standard)
auroc_fp32 = roc_auc_score(y_true, y_scores_fp32)
auroc_int8 = roc_auc_score(y_true, y_scores_int8)
print("-" * 30)
print(f"FP32 AUROC: {auroc_fp32:.4f}")
print(f"INT8 AUROC: {auroc_int8:.4f}")

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.kdeplot(id_scores_fp32, fill=True, label='Malaria', color='blue')
sns.kdeplot(ood_scores_fp32, fill=True, label='WBC (Artifact)', color='red')
plt.axvline(thresh_fp32, color='k', linestyle='--', label='95% TPR Thresh')
plt.title(f'FP32 Reliability (FPR95={fpr95_fp32:.3f})')
plt.legend()

plt.subplot(1, 2, 2)
sns.kdeplot(id_scores_int8, fill=True, label='Malaria', color='blue')
sns.kdeplot(ood_scores_int8, fill=True, label='WBC (Artifact)', color='red')
plt.axvline(thresh_int8, color='k', linestyle='--', label='95% TPR Thresh')
plt.title(f'INT8 Reliability (FPR95={fpr95_int8:.3f})')
plt.legend()

plt.tight_layout()
plt.savefig('clinical_reliability_wbc_stats1.pdf')
print("✅ Plot saved as 'clinical_reliability_wbc_stats1.pdf'")

# --- Step 1: Install & Import Dependencies ---
import subprocess
import sys
import os
import tracemalloc

# Auto-install script for Kaggle or local environments
def install_packages():
    print("Installing missing packages (fvcore, psutil, timm)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "fvcore", "psutil", "timm"])

try:
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    import psutil
    import timm
except ImportError:
    install_packages()
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    import psutil
    import timm

import torch
import torch.nn as nn

# --- Step 2: Setup ---
# We use CPU because we are simulating an edge device (like a smartphone)
DEVICE = torch.device("cpu")
DUMMY_INPUT = torch.randn(1, 3, 224, 224).to(DEVICE)

print("1. Loading Models...")

# 1. Swin FP32
swin_fp32 = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=2)
# swin_fp32.load_state_dict(torch.load("best_swin_malaria.pth", map_location='cpu')) # Optional for this test
swin_fp32.eval()

# 2. Swin INT8
swin_int8 = torch.quantization.quantize_dynamic(
    swin_fp32, {nn.Linear}, dtype=torch.qint8
)
swin_int8.eval()

# 3. MobileNet FP32
mob_fp32 = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=2)
mob_fp32.eval()

# --- Step 3: Computational Complexity (MACs / FLOPs) ---
print("\n" + "="*40)
print("📊 COMPUTATIONAL COMPLEXITY (MACs)")
print("="*40)

def measure_macs(model, name):
    # fvcore counts Multiply-Accumulate Operations (MACs)
    # 1 MAC is roughly equal to 2 FLOPs
    flops = FlopCountAnalysis(model, DUMMY_INPUT)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)

    total_macs = flops.total()
    gmacs = total_macs / 1e9
    print(f"{name:<15}: {gmacs:.3f} GMACs ({total_macs:,} operations)")
    return gmacs

# Note: Quantization changes the *precision* of the operations (FP32 to INT8),
# but the *number* of mathematical operations remains exactly the same.
# Therefore, we only need to measure the FP32 models for this architectural metric.
swin_macs = measure_macs(swin_fp32, "Swin-Tiny")
mob_macs = measure_macs(mob_fp32, "MobileNetV3")

print(f"\nConclusion: Swin-Tiny requires {swin_macs/mob_macs:.1f}x more mathematical operations than MobileNet.")

# --- Step 4: Peak RAM Usage (Inference Memory Footprint) ---
print("\n" + "="*40)
print("🧠 PEAK RAM USAGE (Inference Footprint)")
print("="*40)

def measure_peak_ram(model, name):
    # Force Python garbage collection before measuring
    import gc
    gc.collect()

    # Start tracing Python memory allocations
    tracemalloc.start()

    # Warm-up pass (allocates initial buffers)
    with torch.no_grad():
        _ = model(DUMMY_INPUT)

    # Reset tracing for the actual measurement
    tracemalloc.clear_traces()

    # Benchmark pass
    with torch.no_grad():
        _ = model(DUMMY_INPUT)

    # Get peak memory used during that forward pass
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)
    print(f"{name:<15}: {peak_mb:.2f} MB of RAM allocated during inference")
    return peak_mb

ram_swin_32 = measure_peak_ram(swin_fp32, "Swin FP32")
ram_swin_8 = measure_peak_ram(swin_int8, "Swin INT8")
ram_mob_32 = measure_peak_ram(mob_fp32, "MobileNet FP32")

print("\nSummary for Paper Table:")
print(f"Swin INT8 reduces peak runtime RAM by {(ram_swin_32 - ram_swin_8):.2f} MB compared to its FP32 counterpart.")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ==========================================
# Data Compilation (From the experiments)
# ==========================================
models = ['MobileNet (FP32)', 'MobileNet (INT8)', 'Swin-Tiny (FP32)', 'Swin-Tiny (INT8)']
latency = [11.69, 11.06, 104.15, 103.46] # ms
size_mb = [6.21, 6.21, 110.14, 27.89]    # MB
macs = [0.058, 0.058, 4.51, 4.51]        # GMACs
pgd_acc = [0.0, 0.0, 15.0, 68.0]         # % (Approx FP32 Swin drop vs INT8 68%)

# ==========================================
# Formatting for IEEE Standards
# ==========================================
# IEEE prefers serif fonts and high-contrast readable charts
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'figure.dpi': 300 # Publication quality
})

# ==========================================
# Figure 1: Hardware Footprint (Dual Axis Bar Chart)
# ==========================================
fig, ax1 = plt.subplots(figsize=(10, 6))

x = np.arange(len(models))
width = 0.35

# Plot 1: Latency (Left Axis)
rects1 = ax1.bar(x - width/2, latency, width, label='Latency (ms)', color='#2b8cbe', edgecolor='black')
ax1.set_ylabel('CPU Inference Latency (ms)', color='#2b8cbe', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='#2b8cbe')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=15, ha="right")

# Plot 2: Storage Size (Right Axis)
ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, size_mb, width, label='Storage Size (MB)', color='#e34a33', edgecolor='black')
ax2.set_ylabel('Model Storage Footprint (MB)', color='#e34a33', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#e34a33')

# Title and Legends
plt.title('Hardware Benchmarks: Latency vs. Storage (FP32 vs INT8)', pad=20)
fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.85))

# Add value labels on top of bars
for r in rects1: ax1.annotate(f'{r.get_height():.1f}', (r.get_x() + r.get_width() / 2., r.get_height()), ha='center', va='bottom', fontsize=10)
for r in rects2: ax2.annotate(f'{r.get_height():.1f}', (r.get_x() + r.get_width() / 2., r.get_height()), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('Figure_1_Hardware_Footprint.pdf', format='pdf', dpi=300)
plt.show()

# ==========================================
# Figure 2: The Trade-off Bubble Chart
# ==========================================
fig, ax = plt.subplots(figsize=(9, 6))

# Colors for different architectures
colors = ['#de2d26', '#de2d26', '#3182bd', '#31a354']
# Red for MobileNets (Vulnerable), Blue for Swin FP32, Green for Swin INT8 (Secure)

# Plot Bubbles (Size of bubble = Size in MB * 20 for scaling)
scatter = ax.scatter(macs, pgd_acc, s=[s*25 for s in size_mb], c=colors, alpha=0.7, edgecolors='black', linewidth=2)

# Annotations
for i, txt in enumerate(models):
    # Adjust text position slightly for readability
    y_offset = -5 if 'MobileNet' in txt else 5
    ax.annotate(txt, (macs[i], pgd_acc[i]), xytext=(0, y_offset), textcoords='offset points', ha='center', fontweight='bold')

# Axis formatting
ax.set_xlabel('Computational Complexity (GMACs)', fontweight='bold')
ax.set_ylabel('Adversarial Robustness (PGD Accuracy %)', fontweight='bold')
ax.set_title('The Security-Efficiency Trade-off', pad=15)
ax.grid(True, linestyle='--', alpha=0.6)

# Add a custom legend for bubble size
import matplotlib.lines as mlines
legend_bubbles = [mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=np.sqrt(10*25), label='10 MB'),
                  mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=np.sqrt(50*25), label='50 MB'),
                  mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=np.sqrt(100*25), label='100 MB')]
ax.legend(handles=legend_bubbles, title="Storage Size (MB)", loc='upper left', borderpad=1.5)

# Styling tweaks
ax.set_ylim(-10, 80)
ax.set_xlim(-0.5, 5.5)

plt.tight_layout()
plt.savefig('Figure_2_Security_Tradeoff.pdf', format='pdf', dpi=300)
plt.show()

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe

# IEEE Standard Formatting
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'figure.dpi': 300
})

fig, ax = plt.subplots(figsize=(15, 8))
ax.set_xlim(0, 120)
ax.set_ylim(0, 100)
ax.axis('off')

# Shadow effect for professional depth
shadow = [pe.SimplePatchShadow(offset=(3, -3), shadow_rgbFace='black', alpha=0.15), pe.Normal()]

def draw_modern_box(x, y, width, height, title, text, facecolor, edgecolor):
    # Main Box
    box = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=1.5,rounding_size=3",
        linewidth=2, edgecolor=edgecolor, facecolor=facecolor,
        path_effects=shadow
    )
    ax.add_patch(box)

    # Title Background Ribbon (creates a distinct header area)
    header = patches.FancyBboxPatch(
        (x, y + height - 8), width, 8,
        boxstyle="round,pad=1.5,rounding_size=3",
        linewidth=0, facecolor=edgecolor, alpha=0.1
    )
    # Clip the bottom part of the header so it's a flat line connecting to the box
    header_rect = patches.Rectangle((x, y + height - 8), width, 6, facecolor=edgecolor, alpha=0.1, linewidth=0)
    ax.add_patch(header)
    ax.add_patch(header_rect)

    # Title Text
    ax.text(x + width/2, y + height - 4, title,
            ha='center', va='center', fontweight='bold', fontsize=13, color=edgecolor)

    # Line separator
    ax.plot([x + 2, x + width - 2], [y + height - 8, y + height - 8],
            color=edgecolor, lw=1.5, alpha=0.8)

    # Body text
    ax.text(x + width/2, y + (height - 8)/2, text,
            ha='center', va='center', fontsize=12, linespacing=1.8, color='#333333')

# --- Draw the Blocks ---

# Block 1: Training (Sleek Blue)
text1 = "Dataset: 27,558 Malaria Smears\nArchitecture: Swin-Tiny\nPrecision: 32-bit Floating Point\nSize: 110.14 MB\nClean Accuracy: 98.01%"
draw_modern_box(5, 35, 27, 45, "1. Cloud Server Training", text1, '#f0f8ff', '#0055a4')

# Block 2: Quantization (Sleek Purple)
text2 = "Conversion: FP32 $\\rightarrow$ INT8\nMath: $X_{int8} = \\text{round}(X_f/S) + Z$\nCompression: 3.9x Reduction\nTarget: Mobile Edge Devices"
draw_modern_box(42, 35, 27, 45, "2. Post-Training Quantization", text2, '#f9f0ff', '#5e00a4')

# Block 3: Safe Edge (Sleek Green)
text3 = "Latency: 103.46 ms\nPeak RAM: 0.01 MB\nOOD Safety (AUROC): 0.922\nRobust to Natural Blur/Noise"
draw_modern_box(80, 58, 27, 30, "3A. Clinical Edge Inference", text3, '#f0fff0', '#007a00')

# Block 4: Threat Edge (Sleek Red)
text4 = "Threat: 20-Step PGD Attack\nMechanism: Gradient Masking\nAccuracy Drop: $\\rightarrow$ 0.00%\nResult: Total Model Collapse"
draw_modern_box(80, 20, 27, 30, "3B. Adversarial Security Audit", text4, '#fff0f0', '#a40000')

# --- Draw Curved Connecting Arrows ---
def draw_curved_arrow(xy_start, xy_end, rad, color='#444444'):
    arrow = patches.FancyArrowPatch(
        xy_start, xy_end,
        connectionstyle=f"arc3,rad={rad}",
        color=color,
        arrowstyle="Simple, tail_width=2.5, head_width=10, head_length=12",
        linewidth=2,
        path_effects=[pe.SimplePatchShadow(offset=(2, -2), shadow_rgbFace='black', alpha=0.1), pe.Normal()]
    )
    ax.add_patch(arrow)

# Arrow 1: Training to PTQ (Straight)
draw_curved_arrow((32, 57.5), (42, 57.5), rad=0.0)

# Arrow 2: PTQ to Clinical Inference (Curved Up)
draw_curved_arrow((69, 65), (80, 73), rad=0.2)

# Arrow 3: PTQ to Security Audit (Curved Down)
draw_curved_arrow((69, 50), (80, 35), rad=-0.2)

plt.savefig('Figure_0_Overview_Pro.png', format='pdf', dpi=300, bbox_inches='tight')
print("✅ Pro Overview Figure generated successfully as 'Figure_0_Overview_Pro.pdf'")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe

# IEEE Standard Formatting
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'figure.dpi': 300
})

fig, ax = plt.subplots(figsize=(15, 8))
# Set a soft, modern UI background color
fig.patch.set_facecolor('#f4f6f9')
ax.set_facecolor('#f4f6f9')
ax.set_xlim(0, 120)
ax.set_ylim(0, 100)
ax.axis('off')

def draw_modern_card(x, y, width, height, title, text, theme_color, icon_type):
    # Shadow Effect
    shadow = [pe.SimplePatchShadow(offset=(3, -3), shadow_rgbFace='#000000', alpha=0.08), pe.Normal()]

    # Main White Card
    box = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=1.5,rounding_size=3",
        linewidth=2, edgecolor=theme_color, facecolor='#ffffff',
        path_effects=shadow, zorder=2
    )
    ax.add_patch(box)

    # Colored Header Ribbon (Top part of the card)
    header_height = 14
    header = patches.FancyBboxPatch(
        (x, y + height - header_height), width, header_height,
        boxstyle="round,pad=1.5,rounding_size=3",
        linewidth=0, facecolor=theme_color, alpha=0.1, zorder=3
    )
    # Square off the bottom of the ribbon
    header_rect = patches.Rectangle((x, y + height - header_height), width, 6, facecolor=theme_color, alpha=0.1, linewidth=0, zorder=3)
    ax.add_patch(header)
    ax.add_patch(header_rect)

    # Line separator under header
    ax.plot([x + 1, x + width - 1], [y + height - header_height, y + height - header_height],
            color=theme_color, lw=1.5, alpha=0.4, zorder=4)

    # ---------------------------------------------
    # DRAWING THE CIRCULAR ICON BADGE (Top Center)
    # ---------------------------------------------
    cx, cy = x + width / 2, y + height
    r = 3.5
    badge = patches.Circle((cx, cy), r, facecolor='#ffffff', edgecolor=theme_color, linewidth=2, zorder=10)
    ax.add_patch(badge)

    # Vector Icons Inside Badge
    if icon_type == 'cloud':
        ax.add_patch(patches.Circle((cx - 1.2, cy - 0.4), 1.2, color=theme_color, zorder=11))
        ax.add_patch(patches.Circle((cx + 1.2, cy - 0.4), 1.2, color=theme_color, zorder=11))
        ax.add_patch(patches.Circle((cx, cy + 0.8), 1.6, color=theme_color, zorder=11))
        ax.add_patch(patches.Rectangle((cx - 1.2, cy - 1.6), 2.4, 1.6, color=theme_color, zorder=11))
    elif icon_type == 'compress':
        # Large matrix box
        ax.add_patch(patches.Rectangle((cx - 1.6, cy + 0.4), 3.2, 1.4, facecolor='none', edgecolor=theme_color, lw=2, zorder=11))
        # Small matrix box
        ax.add_patch(patches.Rectangle((cx - 0.8, cy - 1.8), 1.6, 0.8, facecolor='none', edgecolor=theme_color, lw=2, zorder=11))
        # Arrow down
        ax.add_patch(patches.FancyArrowPatch((cx, cy + 0.1), (cx, cy - 0.7), color=theme_color, arrowstyle="-|>,head_length=3.5,head_width=3.5", lw=2, zorder=11))
    elif icon_type == 'medical':
        ax.add_patch(patches.Rectangle((cx - 1.8, cy - 0.6), 3.6, 1.2, color=theme_color, zorder=11))
        ax.add_patch(patches.Rectangle((cx - 0.6, cy - 1.8), 1.2, 3.6, color=theme_color, zorder=11))
    elif icon_type == 'shield':
        shield = patches.Polygon([[cx - 1.6, cy + 1.2], [cx + 1.6, cy + 1.2], [cx, cy - 1.8]], facecolor='none', edgecolor=theme_color, lw=2.2, zorder=11)
        ax.add_patch(shield)
        # Exclamation point
        ax.plot([cx, cx], [cy + 0.5, cy - 0.2], color=theme_color, lw=2.2, zorder=11)
        ax.add_patch(patches.Circle((cx, cy - 0.8), 0.25, color=theme_color, zorder=11))

    # ---------------------------------------------
    # TEXT PLACEMENT
    # ---------------------------------------------
    # Title
    ax.text(cx, y + height - 7, title,
            ha='center', va='center', fontweight='bold', fontsize=13, color=theme_color, zorder=4)

    # Body text
    ax.text(cx, y + (height - header_height)/2, text,
            ha='center', va='center', fontsize=12, linespacing=1.8, color='#222222', zorder=4)

def draw_curved_arrow(xy_start, xy_end, rad, color='#999999'):
    arrow = patches.FancyArrowPatch(
        xy_start, xy_end,
        connectionstyle=f"arc3,rad={rad}",
        color=color,
        arrowstyle="Simple, tail_width=2.5, head_width=10, head_length=12",
        linewidth=1,
        path_effects=[pe.SimplePatchShadow(offset=(2, -2), shadow_rgbFace='#000000', alpha=0.08), pe.Normal()],
        zorder=1
    )
    ax.add_patch(arrow)

# --- Draw the Cards ---
t1 = "Dataset: 27,558 Malaria Smears\nArchitecture: Swin-Tiny\nPrecision: 32-bit Floating Point\nSize: 110.14 MB\nClean Accuracy: 98.01%"
draw_modern_card(5, 33, 27, 44, "1. Cloud Server Training", t1, '#005b9f', 'cloud')

t2 = "Conversion: FP32 $\\rightarrow$ INT8\nMath: $X_{int8} = \\text{round}(X_f/S) + Z$\nCompression: 3.9x Reduction\nTarget: Mobile Edge Devices"
draw_modern_card(42, 33, 27, 44, "2. PTQ Pipeline", t2, '#6a0dad', 'compress')

t3 = "Latency: 103.46 ms\nPeak RAM: 0.01 MB\nOOD Safety (AUROC): 0.922\nRobust to Natural Blur/Noise"
draw_modern_card(80, 56, 27, 30, "3A. Clinical Edge Inference", t3, '#1e7b1e', 'medical')

t4 = "Threat: 20-Step PGD Attack\nMechanism: Gradient Masking\nAccuracy Drop: $\\rightarrow$ 0.00%\nResult: Total Model Collapse"
draw_modern_card(80, 18, 27, 30, "3B. Adversarial Security Audit", t4, '#b30000', 'shield')

# --- Draw Connectors ---
draw_curved_arrow((32, 55), (42, 55), rad=0.0)
draw_curved_arrow((69, 62), (80, 71), rad=0.2)
draw_curved_arrow((69, 48), (80, 33), rad=-0.2)

# Save figure with tight bounding box to wrap the padded areas nicely
plt.savefig('Figure_0_Overview_Pro.png', format='pdf', dpi=300, bbox_inches='tight')
print("✅ Pro Infographic generated successfully as 'Figure_0_Overview_Pro.pdf'")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe

# IEEE Standard Formatting
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'figure.dpi': 300
})

fig, ax = plt.subplots(figsize=(15, 8))
# Set a soft, modern UI background color
fig.patch.set_facecolor('#f4f6f9')
ax.set_facecolor('#f4f6f9')
ax.set_xlim(0, 120)
ax.set_ylim(0, 100)
ax.axis('off')

def draw_modern_card(x, y, width, height, title, text, theme_color, icon_type):
    # Shadow Effect
    shadow = [pe.SimplePatchShadow(offset=(3, -3), shadow_rgbFace='#000000', alpha=0.08), pe.Normal()]

    # Main White Card
    box = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=1.5,rounding_size=3",
        linewidth=2, edgecolor=theme_color, facecolor='#ffffff',
        path_effects=shadow, zorder=2
    )
    ax.add_patch(box)

    # Colored Header Ribbon (Top part of the card)
    header_height = 14
    header = patches.FancyBboxPatch(
        (x, y + height - header_height), width, header_height,
        boxstyle="round,pad=1.5,rounding_size=3",
        linewidth=0, facecolor=theme_color, alpha=0.1, zorder=3
    )
    # Square off the bottom of the ribbon
    header_rect = patches.Rectangle((x, y + height - header_height), width, 6, facecolor=theme_color, alpha=0.1, linewidth=0, zorder=3)
    ax.add_patch(header)
    ax.add_patch(header_rect)

    # Line separator under header
    ax.plot([x + 1, x + width - 1], [y + height - header_height, y + height - header_height],
            color=theme_color, lw=1.5, alpha=0.4, zorder=4)

    # ---------------------------------------------
    # DRAWING THE CIRCULAR ICON BADGE (Top Center)
    # ---------------------------------------------
    cx, cy = x + width / 2, y + height
    r = 3.5
    badge = patches.Circle((cx, cy), r, facecolor='#ffffff', edgecolor=theme_color, linewidth=2, zorder=10)
    ax.add_patch(badge)

    # Vector Icons Inside Badge
    if icon_type == 'cloud':
        ax.add_patch(patches.Circle((cx - 1.2, cy - 0.4), 1.2, color=theme_color, zorder=11))
        ax.add_patch(patches.Circle((cx + 1.2, cy - 0.4), 1.2, color=theme_color, zorder=11))
        ax.add_patch(patches.Circle((cx, cy + 0.8), 1.6, color=theme_color, zorder=11))
        ax.add_patch(patches.Rectangle((cx - 1.2, cy - 1.6), 2.4, 1.6, color=theme_color, zorder=11))
    elif icon_type == 'compress':
        # Large matrix box
        ax.add_patch(patches.Rectangle((cx - 1.6, cy + 0.4), 3.2, 1.4, facecolor='none', edgecolor=theme_color, lw=2, zorder=11))
        # Small matrix box
        ax.add_patch(patches.Rectangle((cx - 0.8, cy - 1.8), 1.6, 0.8, facecolor='none', edgecolor=theme_color, lw=2, zorder=11))
        # Arrow down
        ax.add_patch(patches.FancyArrowPatch((cx, cy + 0.1), (cx, cy - 0.7), color=theme_color, arrowstyle="-|>,head_length=3.5,head_width=3.5", lw=2, zorder=11))
    elif icon_type == 'crescent':
        # Mathematically drawing a crescent using an offset background-colored cutout circle
        ax.add_patch(patches.Circle((cx - 0.2, cy), 1.8, facecolor=theme_color, edgecolor='none', zorder=11))
        ax.add_patch(patches.Circle((cx + 0.5, cy), 1.6, facecolor='#ffffff', edgecolor='none', zorder=12))
    elif icon_type == 'shield':
        shield = patches.Polygon([[cx - 1.6, cy + 1.2], [cx + 1.6, cy + 1.2], [cx, cy - 1.8]], facecolor='none', edgecolor=theme_color, lw=2.2, zorder=11)
        ax.add_patch(shield)
        # Exclamation point
        ax.plot([cx, cx], [cy + 0.5, cy - 0.2], color=theme_color, lw=2.2, zorder=11)
        ax.add_patch(patches.Circle((cx, cy - 0.8), 0.25, color=theme_color, zorder=11))

    # ---------------------------------------------
    # TEXT PLACEMENT
    # ---------------------------------------------
    # Title - Dynamically adjust font size for longer titles to ensure perfect fit
    title_font_size = 12 if len(title) > 28 else 12.5
    ax.text(cx, y + height - 7, title,
            ha='center', va='center', fontweight='bold', fontsize=title_font_size, color=theme_color, zorder=4)

    # Body text
    ax.text(cx, y + (height - header_height)/2, text,
            ha='center', va='center', fontsize=12, linespacing=1.8, color='#222222', zorder=4)

def draw_curved_arrow(xy_start, xy_end, rad, color='#999999'):
    arrow = patches.FancyArrowPatch(
        xy_start, xy_end,
        connectionstyle=f"arc3,rad={rad}",
        color=color,
        arrowstyle="Simple, tail_width=2.5, head_width=10, head_length=12",
        linewidth=1,
        path_effects=[pe.SimplePatchShadow(offset=(2, -2), shadow_rgbFace='#000000', alpha=0.08), pe.Normal()],
        zorder=1
    )
    ax.add_patch(arrow)

# --- Draw the Cards (Adjusted coordinates and expanded widths) ---

t1 = "Dataset: 27,558 Malaria Smears\nArchitecture: Swin-Tiny\nPrecision: 32-bit Floating Point\nSize: 110.14 MB\nClean Accuracy: 98.01%"
draw_modern_card(3, 33, 27, 44, "1. Cloud Server Training", t1, '#005b9f', 'cloud')

t2 = "Conversion: FP32 $\\rightarrow$ INT8\nMath: $X_{int8} = \\text{round}(X_f/S) + Z$\nCompression: 3.9x Reduction\nTarget: Mobile Edge Devices"
draw_modern_card(39, 33, 27, 44, "2. PTQ Pipeline", t2, '#6a0dad', 'compress')

# Width expanded from 27 to 31 to guarantee text fits beautifully
t3 = "Latency: 103.46 ms\nPeak RAM: 0.01 MB\nOOD Safety (AUROC): 0.922\nRobust to Natural Blur/Noise"
draw_modern_card(76, 56, 31, 30, "3A. Clinical Edge Inference", t3, '#1e7b1e', 'crescent')

t4 = "Threat: 20-Step PGD Attack\nMechanism: Gradient Masking\nAccuracy Drop: $\\rightarrow$ 0.00%\nResult: Total Model Collapse"
draw_modern_card(76, 18, 31, 30, "3B. Adversarial Security Audit", t4, '#b30000', 'shield')

# --- Draw Connectors ---
draw_curved_arrow((30, 55), (39, 55), rad=0.0)
draw_curved_arrow((66, 62), (76, 71), rad=0.2)
draw_curved_arrow((66, 48), (76, 33), rad=-0.2)

# Save figure with tight bounding box to wrap the padded areas nicely
plt.savefig('Figure_0_Overview_Pro.png', format='pdf', dpi=300, bbox_inches='tight')
print("✅ Pro Infographic generated successfully as 'Figure_0_Overview_Pro.pdf'")

# --- Step 1: Imports & Setup ---
import torch
import torch.nn as nn
import timm
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.notebook import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU = torch.device("cpu")

# --- Step 2: Dataset Loading (Subset for testing) ---
class SafeMalariaDataset(torch.utils.data.Dataset):
    def __init__(self, p_path, u_path, transform):
        # We use 500 images total for the transfer attack to save time
        self.files = ([os.path.join(p_path, f) for f in os.listdir(p_path) if f.lower().endswith('.png')][:250] +
                      [os.path.join(u_path, f) for f in os.listdir(u_path) if f.lower().endswith('.png')][:250])
        self.labels = [0]*250 + [1]*250
        self.transform = transform

    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        try: return self.transform(Image.open(self.files[i]).convert('RGB')), self.labels[i]
        except: return torch.zeros(3,224,224), self.labels[i]

base_path = '/kaggle/input/cell-images-for-detecting-malaria'
p_dir, u_dir = None, None
for root, dirs, files in os.walk(base_path):
    if 'Parasitized' in dirs: p_dir = os.path.join(root, 'Parasitized')
    if 'Uninfected' in dirs: u_dir = os.path.join(root, 'Uninfected')

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=norm_mean, std=norm_std)
])

test_loader = DataLoader(SafeMalariaDataset(p_dir, u_dir, test_transform), batch_size=16, shuffle=False)

# --- Step 3: PGD Attack Function (Self-contained) ---
def pgd_attack(model, images, labels, eps=0.03, alpha=2/255, steps=20, device='cuda', mean=None, std=None):
    model.eval()
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean).view(1, 3, 1, 1).to(device)
        std_t = torch.tensor(std).view(1, 3, 1, 1).to(device)
        images_raw = images.clone().detach() * std_t + mean_t
        images_raw = torch.clamp(images_raw, 0, 1)
    else:
        mean_t, std_t = None, None
        images_raw = images.clone().detach()

    adv_images_raw = images_raw.clone().detach()
    adv_images_raw.requires_grad = True

    for step in range(steps):
        if mean_t is not None: adv_input = (adv_images_raw - mean_t) / std_t
        else: adv_input = adv_images_raw

        outputs = model(adv_input)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()

        grad = adv_images_raw.grad.data
        adv_images_raw.data = adv_images_raw.data + alpha * grad.sign()
        eta = torch.clamp(adv_images_raw.data - images_raw, min=-eps, max=eps)
        adv_images_raw.data = torch.clamp(images_raw + eta, min=0, max=1)
        adv_images_raw.grad = None

    if mean_t is not None: return (adv_images_raw.detach() - mean_t) / std_t
    else: return adv_images_raw.detach()

# --- Step 4: Load Models ---
print("\n" + "="*40)
print("🥷 BLACK-BOX TRANSFER ATTACK AUDIT")
print("="*40)

print("1️⃣ Loading Surrogate Model (Attacker's CNN)...")
surrogate_model = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=2)
surrogate_model.load_state_dict(torch.load("mobilenet_temp.pth", map_location=DEVICE))
surrogate_model.to(DEVICE).eval()

print("2️⃣ Loading Target Models (Victim's Transformers)...")
target_fp32 = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=2)
target_fp32.load_state_dict(torch.load("/kaggle/working/best_swin_malaria.pth", map_location=DEVICE))
target_int8 = torch.quantization.quantize_dynamic(target_fp32, {nn.Linear}, dtype=torch.qint8)

target_fp32.to(DEVICE).eval()
target_int8.to(CPU).eval() # INT8 must run on CPU

# --- Step 5: The Attack Loop ---
epsilons = [0, 0.01, 0.03, 0.05, 0.1]
results = {'eps': epsilons, 'surrogate_wb': [], 'target_fp32_bb': [], 'target_int8_bb': []}

print("3️⃣ Launching Attacks (Source: MobileNet -> Target: Swin)...")

for eps in epsilons:
    correct_surrogate = 0
    correct_target_32 = 0
    correct_target_8 = 0
    total = 0

    for images, labels in tqdm(test_loader, desc=f"Eps {eps}", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        total += labels.size(0)

        # 1. Generate Attack on the Surrogate (MobileNetV3)
        if eps == 0:
            adv_images = images
        else:
            adv_images = pgd_attack(surrogate_model, images, labels, eps=eps, steps=20, device=DEVICE, mean=norm_mean, std=norm_std)

        # 2. Evaluate Surrogate (White-box success rate)
        with torch.no_grad():
            pred_surr = surrogate_model(adv_images).argmax(1)
            correct_surrogate += (pred_surr == labels).sum().item()

            # 3. Evaluate Target FP32 (Black-box transfer)
            pred_t32 = target_fp32(adv_images).argmax(1)
            correct_target_32 += (pred_t32 == labels).sum().item()

            # 4. Evaluate Target INT8 (Black-box transfer on Edge)
            pred_t8 = target_int8(adv_images.cpu()).argmax(1).to(DEVICE)
            correct_target_8 += (pred_t8 == labels).sum().item()

    acc_surr = correct_surrogate / total
    acc_t32 = correct_target_32 / total
    acc_t8 = correct_target_8 / total

    results['surrogate_wb'].append(acc_surr)
    results['target_fp32_bb'].append(acc_t32)
    results['target_int8_bb'].append(acc_t8)

    print(f"   [Eps {eps:<4}] Surrogate(WB): {acc_surr:.2f} | Target FP32(BB): {acc_t32:.2f} | Target INT8(BB): {acc_t8:.2f}")

# --- Step 6: Visualization for the Paper ---
plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'figure.dpi': 300})
plt.figure(figsize=(8, 6))

plt.plot(results['eps'], results['surrogate_wb'], 'k--x', alpha=0.5, label='MobileNet FP32 (White-Box Source)')
plt.plot(results['eps'], results['target_fp32_bb'], 'b-o', linewidth=2, label='Swin FP32 (Black-Box Transfer)')
plt.plot(results['eps'], results['target_int8_bb'], 'g-s', linewidth=2, label='Swin INT8 (Black-Box Transfer)')

plt.title("Black-Box Transfer Attack Robustness\n(Source: MobileNetV3 $\\rightarrow$ Target: Swin-Tiny)")
plt.xlabel("Attack Strength (Epsilon)")
plt.ylabel("Model Accuracy")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('Figure_BlackBox_Transfer.pdf', format='pdf', dpi=300)
print("\n✅ Black-Box Audit Complete. Plot saved as 'Figure_BlackBox_Transfer.pdf'")

# -*- coding: utf-8 -*-
"""Bootstrap 95% CI Calculator for Malaria Models"""

import torch
import torch.nn as nn
import numpy as np
import timm
import os
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.utils import resample

# --- Step 1: Setup and Reproducibility ---
DEVICE_GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_CPU = torch.device("cpu")

def seed_everything(seed=42):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# --- Step 2: Recreate the Exact Test Dataset ---
class SafeMalariaDataset(torch.utils.data.Dataset):
    def __init__(self, parasitized_path, uninfected_path, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        
        p_files = [os.path.join(parasitized_path, f) for f in os.listdir(parasitized_path) if f.lower().endswith('.png')]
        u_files = [os.path.join(uninfected_path, f) for f in os.listdir(uninfected_path) if f.lower().endswith('.png')]
        
        self.image_paths.extend(p_files)
        self.labels.extend([0] * len(p_files))
        self.image_paths.extend(u_files)
        self.labels.extend([1] * len(u_files))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except Exception:
            return torch.zeros((3, 224, 224)), self.labels[idx]

base_path = '/kaggle/input/cell-images-for-detecting-malaria'
p_dir, u_dir = None, None
for root, dirs, files in os.walk(base_path):
    if os.path.basename(root) == 'Parasitized': p_dir = root
    elif os.path.basename(root) == 'Uninfected': u_dir = root

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_dataset = SafeMalariaDataset(p_dir, u_dir, transform=None)
total_size = len(full_dataset)

train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

generator = torch.Generator().manual_seed(42)
_, _, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=generator)

class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return self.transform(x), y
    def __len__(self): return len(self.subset)

test_set_final = TransformedSubset(test_dataset, eval_transform)
test_loader = DataLoader(test_set_final, batch_size=64, shuffle=False, num_workers=2)

# --- Step 3: Load Models ---
print("Loading Swin-Tiny models...")
swin_fp32 = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=2)
swin_fp32.load_state_dict(torch.load("/kaggle/working/best_swin_malaria.pth", map_location='cpu'))
swin_fp32.to(DEVICE_GPU).eval()

swin_int8 = torch.quantization.quantize_dynamic(swin_fp32.to(DEVICE_CPU), {nn.Linear}, dtype=torch.qint8)
swin_int8.eval()
swin_fp32.to(DEVICE_GPU) # Move FP32 back to GPU

print("Loading MobileNetV3 models...")
mob_fp32 = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=2)

mob_fp32.load_state_dict(torch.load("mobilenet_temp.pth", map_location='cpu'))
mob_fp32.to(DEVICE_GPU).eval()

mob_int8 = torch.quantization.quantize_dynamic(mob_fp32.to(DEVICE_CPU), {nn.Linear}, dtype=torch.qint8)
mob_int8.eval()
mob_fp32.to(DEVICE_GPU)

# --- Step 4: Extract Predictions ---
print("Extracting predictions for all models...")
true_labels = []
preds_swin_32, preds_swin_8 = [], []
preds_mob_32, preds_mob_8 = [], []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Inference"):
        true_labels.extend(labels.numpy())
        
        # FP32 on GPU
        inputs_gpu = inputs.to(DEVICE_GPU)
        preds_swin_32.extend(swin_fp32(inputs_gpu).argmax(1).cpu().numpy())
        preds_mob_32.extend(mob_fp32(inputs_gpu).argmax(1).cpu().numpy())
        
        # INT8 on CPU
        preds_swin_8.extend(swin_int8(inputs).argmax(1).numpy())
        preds_mob_8.extend(mob_int8(inputs).argmax(1).numpy())

true_labels = np.array(true_labels)
predictions = {
    "Swin-Tiny FP32": np.array(preds_swin_32),
    "Swin-Tiny INT8": np.array(preds_swin_8),
    "MobileNetV3 FP32": np.array(preds_mob_32),
    "MobileNetV3 INT8": np.array(preds_mob_8)
}

# --- Step 5: Bootstrapping for 95% CI ---
print("\nCalculating 95% Confidence Intervals via Bootstrapping (1000 iterations)...")
n_iterations = 1000

for model_name, preds in predictions.items():
    bootstrapped_accuracies = []
    
    for i in range(n_iterations):
        # Resample indices with replacement
        indices = resample(np.arange(len(true_labels)), replace=True, random_state=i)
        
        # Calculate accuracy for this resampled subset
        resampled_true = true_labels[indices]
        resampled_preds = preds[indices]
        acc = np.mean(resampled_true == resampled_preds)
        bootstrapped_accuracies.append(acc)
    
    # Calculate 2.5th and 97.5th percentiles
    lower_bound = np.percentile(bootstrapped_accuracies, 2.5) * 100
    upper_bound = np.percentile(bootstrapped_accuracies, 97.5) * 100
    mean_acc = np.mean(true_labels == preds) * 100
    
    print(f"{model_name:<18}: {mean_acc:.2f}% (95% CI: {lower_bound:.2f}% - {upper_bound:.2f}%)")
