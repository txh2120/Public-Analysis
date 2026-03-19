"""
Medical MNIST Classification - Full Training Pipeline
Run this from a Kaggle notebook that has the medical-mnist dataset attached:
    !git clone https://github.com/txh2120/Public-Analysis.git
    %run Public-Analysis/src/medical_mnist_train.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================
# 1. Setup
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
torch.manual_seed(42)
np.random.seed(42)

# Auto-detect dataset path
POSSIBLE_PATHS = [
    '/kaggle/input/medical-mnist',
    '/kaggle/input/datasets/andrewmvd/medical-mnist',
    '/kaggle/input/andrewmvd/medical-mnist',
]
DATA_DIR = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p):
        DATA_DIR = p
        break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Dataset not found. Create this notebook from the medical-mnist dataset page "
        "so the data is automatically attached."
    )

print(f'Using DATA_DIR: {DATA_DIR}')

# ============================================================
# 2. EDA
# ============================================================
classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
print(f'Classes ({len(classes)}): {classes}')

class_counts = {}
for cls in classes:
    cls_path = os.path.join(DATA_DIR, cls)
    count = len(os.listdir(cls_path))
    class_counts[cls] = count
    print(f'  {cls}: {count} images')
print(f'\nTotal images: {sum(class_counts.values())}')

# Class distribution plot
plt.figure(figsize=(10, 5))
bars = plt.bar(class_counts.keys(), class_counts.values(), color='steelblue')
plt.title('Class Distribution', fontsize=14)
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)
for bar, count in zip(bars, class_counts.values()):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 100,
             str(count), ha='center', va='bottom', fontsize=11)
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=100)
plt.show()

# Sample images
fig, axes = plt.subplots(2, len(classes), figsize=(18, 6))
fig.suptitle('Sample Images per Class', fontsize=14)
for col, cls in enumerate(classes):
    cls_path = os.path.join(DATA_DIR, cls)
    img_files = os.listdir(cls_path)[:2]
    for row, img_file in enumerate(img_files):
        img = Image.open(os.path.join(cls_path, img_file))
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].axis('off')
        if row == 0:
            axes[row, col].set_title(cls, fontsize=11)
plt.tight_layout()
plt.savefig('sample_images.png', dpi=100)
plt.show()

sample_img = Image.open(os.path.join(DATA_DIR, classes[0], os.listdir(os.path.join(DATA_DIR, classes[0]))[0]))
print(f'Image size: {sample_img.size}, Mode: {sample_img.mode}')

# ============================================================
# 3. Dataset & DataLoader
# ============================================================
class MedicalMNISTDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            cls_path = os.path.join(data_dir, cls)
            for img_name in os.listdir(cls_path):
                self.samples.append((os.path.join(cls_path, img_name), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

full_dataset = MedicalMNISTDataset(DATA_DIR, transform=transform)
print(f'Total dataset size: {len(full_dataset)}')
print(f'Class mapping: {full_dataset.class_to_idx}')

total = len(full_dataset)
train_size = int(0.7 * total)
val_size = int(0.15 * total)
test_size = total - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)
print(f'Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}')

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ============================================================
# 4. CNN Model
# ============================================================
class MedicalCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(MedicalCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = MedicalCNN(num_classes=len(classes)).to(device)
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f'\nTotal params: {total_params:,}')

# ============================================================
# 5. Training
# ============================================================
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
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
        torch.save(model.state_dict(), 'best_model.pth')
    print(f'Epoch [{epoch+1:2d}/{NUM_EPOCHS}] '
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
# 6. Evaluation
# ============================================================
model.load_state_dict(torch.load('best_model.pth'))
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f'\n===== Test Results =====')
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)')

# Confusion matrix
all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix', fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100)
plt.show()

print('\n===== Classification Report =====')
print(classification_report(all_labels, all_preds, target_names=classes))

# Sample predictions
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
indices = np.random.choice(len(test_dataset), 16, replace=False)
model.eval()
with torch.no_grad():
    for i, idx in enumerate(indices):
        image, label = test_dataset[idx]
        output = model(image.unsqueeze(0).to(device))
        _, predicted = output.max(1)
        pred_idx = predicted.item()
        ax = axes[i // 4, i % 4]
        img_display = image.squeeze().numpy() * 0.5 + 0.5
        ax.imshow(img_display, cmap='gray')
        actual = classes[label]
        pred = classes[pred_idx]
        color = 'green' if label == pred_idx else 'red'
        ax.set_title(f'Pred: {pred}\nActual: {actual}', color=color, fontsize=9)
        ax.axis('off')
plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', fontsize=14)
plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=100)
plt.show()

print('\nDone! All plots saved.')
