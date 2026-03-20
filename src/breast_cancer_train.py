"""
Breast Cancer Histopathology IDC Classification - Transfer Learning Pipeline
Run this from a Kaggle notebook that has the breast-histopathology-images dataset attached:
    !git clone https://github.com/txh2120/Public-Analysis.git
    %run Public-Analysis/src/breast_cancer_train.py

Binary classification: IDC positive vs negative using ResNet18 pretrained.
"""

import os
import glob
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from torchvision import transforms, models
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, auc
)

# ============================================================
# 1. Setup
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Auto-detect dataset path
POSSIBLE_PATHS = [
    '/kaggle/input/breast-histopathology-images',
    '/kaggle/input/datasets/paultimothymooney/breast-histopathology-images',
    '/kaggle/input/paultimothymooney/breast-histopathology-images',
]
DATA_DIR = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p):
        DATA_DIR = p
        break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Dataset not found. Create this notebook from the breast-histopathology-images "
        "dataset page so the data is automatically attached."
    )

print(f'Using DATA_DIR: {DATA_DIR}')

# ============================================================
# 2. Collect Image Paths
# ============================================================
# The dataset structure: DATA_DIR/<patient_id>/<class_label>/image.png
# class 0 = IDC negative, class 1 = IDC positive
# Collect all images using glob patterns for both class directories

all_image_paths = []
all_labels = []

for label in [0, 1]:
    pattern = os.path.join(DATA_DIR, '**', str(label), '*.png')
    paths = glob.glob(pattern, recursive=True)
    all_image_paths.extend(paths)
    all_labels.extend([label] * len(paths))

print(f'Total images found: {len(all_image_paths)}')

CLASS_NAMES = ['IDC Negative (0)', 'IDC Positive (1)']
label_counts = Counter(all_labels)
for label, name in enumerate(CLASS_NAMES):
    print(f'  {name}: {label_counts[label]} images')

imbalance_ratio = label_counts[0] / label_counts[1]
print(f'Imbalance ratio (neg:pos): {imbalance_ratio:.2f}:1')

# ============================================================
# 3. EDA
# ============================================================
# Class distribution plot
plt.figure(figsize=(8, 5))
bars = plt.bar(CLASS_NAMES, [label_counts[0], label_counts[1]],
               color=['steelblue', 'salmon'])
plt.title('Class Distribution: IDC Negative vs Positive', fontsize=14)
plt.xlabel('Class')
plt.ylabel('Number of Images')
for bar, count in zip(bars, [label_counts[0], label_counts[1]]):
    plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 500,
             f'{count:,}', ha='center', va='bottom', fontsize=12)
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=100)
plt.show()

# Sample images from each class
fig, axes = plt.subplots(2, 5, figsize=(15, 7))
fig.suptitle('Sample Images per Class', fontsize=14)
for row, label in enumerate([0, 1]):
    label_indices = [i for i, l in enumerate(all_labels) if l == label]
    sample_indices = random.sample(label_indices, 5)
    for col, idx in enumerate(sample_indices):
        img = Image.open(all_image_paths[idx])
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        if col == 0:
            axes[row, col].set_ylabel(CLASS_NAMES[label], fontsize=11)
    axes[row, 0].set_title(CLASS_NAMES[label], fontsize=10)
plt.tight_layout()
plt.savefig('sample_images.png', dpi=100)
plt.show()

sample_img = Image.open(all_image_paths[0])
print(f'Image size: {sample_img.size}, Mode: {sample_img.mode}')

# ============================================================
# 4. Dataset & DataLoader
# ============================================================
class BreastCancerDataset(Dataset):
    """Custom dataset for breast cancer histopathology patches."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# Data augmentation for training; standard transform for val/test
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Shuffle and split into train/val/test (70/15/15)
combined = list(zip(all_image_paths, all_labels))
random.shuffle(combined)
all_image_paths_shuffled, all_labels_shuffled = zip(*combined)
all_image_paths_shuffled = list(all_image_paths_shuffled)
all_labels_shuffled = list(all_labels_shuffled)

total = len(all_image_paths_shuffled)
train_size = int(0.7 * total)
val_size = int(0.15 * total)
test_size = total - train_size - val_size

train_paths = all_image_paths_shuffled[:train_size]
train_labels = all_labels_shuffled[:train_size]
val_paths = all_image_paths_shuffled[train_size:train_size + val_size]
val_labels = all_labels_shuffled[train_size:train_size + val_size]
test_paths = all_image_paths_shuffled[train_size + val_size:]
test_labels = all_labels_shuffled[train_size + val_size:]

train_dataset = BreastCancerDataset(train_paths, train_labels, transform=train_transform)
val_dataset = BreastCancerDataset(val_paths, val_labels, transform=val_test_transform)
test_dataset = BreastCancerDataset(test_paths, test_labels, transform=val_test_transform)

print(f'Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}')

# WeightedRandomSampler to handle class imbalance during training
train_label_counts = Counter(train_labels)
class_weights = {cls: 1.0 / count for cls, count in train_label_counts.items()}
sample_weights = [class_weights[label] for label in train_labels]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f'Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}')

# ============================================================
# 5. Model - ResNet18 Transfer Learning
# ============================================================
model = models.resnet18(weights='IMAGENET1K_V1')

# Freeze early layers for transfer learning efficiency
for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer4 and the fc layer for fine-tuning
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace the final fully connected layer for binary classification
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 1)
)

model = model.to(device)
print(model)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'\nTotal params: {total_params:,}')
print(f'Trainable params: {trainable_params:,}')

# ============================================================
# 6. Training
# ============================================================
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train the model for one epoch and return average loss and accuracy."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        predicted = (torch.sigmoid(outputs) >= 0.5).long()
        total += labels.size(0)
        correct += predicted.eq(labels.long()).sum().item()
    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate the model and return average loss and accuracy."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().to(device)
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            predicted = (torch.sigmoid(outputs) >= 0.5).long()
            total += labels.size(0)
            correct += predicted.eq(labels.long()).sum().item()
    return running_loss / total, correct / total


print('Training started!')
print('=' * 70)
best_val_acc = 0.0
for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_breast_cancer_model.pth')
    print(f'Epoch [{epoch + 1:2d}/{NUM_EPOCHS}] '
          f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
          f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}'
          f'{" * Best" if val_acc >= best_val_acc else ""}')
print('=' * 70)
print(f'Training complete! Best Validation Accuracy: {best_val_acc:.4f}')

# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
epochs_range = range(1, NUM_EPOCHS + 1)
ax1.plot(epochs_range, history['train_loss'], 'b-o', label='Train Loss')
ax1.plot(epochs_range, history['val_loss'], 'r-o', label='Val Loss')
ax1.set_title('Loss per Epoch')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax2.plot(epochs_range, history['train_acc'], 'b-o', label='Train Acc')
ax2.plot(epochs_range, history['val_acc'], 'r-o', label='Val Acc')
ax2.set_title('Accuracy per Epoch')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_curves.png', dpi=100)
plt.show()

# ============================================================
# 7. Evaluation
# ============================================================
model.load_state_dict(torch.load('best_breast_cancer_model.pth', weights_only=True))
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f'\n===== Test Results =====')
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_acc:.4f} ({test_acc * 100:.1f}%)')

# Collect predictions and probabilities for AUC-ROC
all_preds = []
all_labels_collected = []
all_probs = []
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images).squeeze(1)
        probs = torch.sigmoid(outputs)
        predicted = (probs >= 0.5).long()
        all_preds.extend(predicted.cpu().numpy())
        all_labels_collected.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels_collected = np.array(all_labels_collected)
all_probs = np.array(all_probs)

# AUC-ROC Score
auc_roc = roc_auc_score(all_labels_collected, all_probs)
print(f'\nAUC-ROC Score: {auc_roc:.4f}')
if auc_roc >= 0.85:
    print(f'Target AUC-ROC >= 0.85 ACHIEVED!')
else:
    print(f'Target AUC-ROC >= 0.85 not yet reached. Consider more epochs or tuning.')

# ROC Curve
fpr, tpr, thresholds = roc_curve(all_labels_collected, all_probs)
roc_auc_val = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc_val:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - IDC Classification', fontsize=14)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=100)
plt.show()

# Confusion Matrix
cm = confusion_matrix(all_labels_collected, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix', fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100)
plt.show()

# Classification Report
print('\n===== Classification Report =====')
print(classification_report(all_labels_collected, all_preds,
                            target_names=CLASS_NAMES))

# ============================================================
# 8. Sample Predictions
# ============================================================
fig, axes = plt.subplots(4, 4, figsize=(14, 14))
indices = np.random.choice(len(test_dataset), 16, replace=False)
model.eval()

# Inverse normalization for display
inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225]
)

with torch.no_grad():
    for i, idx in enumerate(indices):
        image, label = test_dataset[idx]
        output = model(image.unsqueeze(0).to(device)).squeeze(1)
        prob = torch.sigmoid(output).item()
        pred_idx = 1 if prob >= 0.5 else 0
        ax = axes[i // 4, i % 4]
        # Inverse normalize for display
        img_display = inv_normalize(image)
        img_display = torch.clamp(img_display, 0, 1)
        img_display = img_display.permute(1, 2, 0).numpy()
        ax.imshow(img_display)
        actual = CLASS_NAMES[label]
        pred = CLASS_NAMES[pred_idx]
        color = 'green' if label == pred_idx else 'red'
        ax.set_title(f'Pred: {pred}\nActual: {actual}\nProb: {prob:.3f}',
                     color=color, fontsize=8)
        ax.axis('off')
plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', fontsize=14)
plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=100)
plt.show()

print('\nDone! All plots saved.')
