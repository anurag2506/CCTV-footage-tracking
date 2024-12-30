import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Device configuration
device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)

torch.manual_seed(42)
np.random.seed(42)

# Data Augmentation
train_transforms = transforms.Compose(
    [
        transforms.Resize((256, 256)),  # Resize the image to a fixed size
        transforms.RandomCrop(224),  # Randomly crop to a fixed size
        transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
        transforms.RandomRotation(45),  # Random rotation
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        ),  # Random color jitter
        transforms.RandomAffine(
            degrees=45, translate=(0.2, 0.2), scale=(0.8, 1.3)
        ),  # Random affine transformations
        transforms.GaussianBlur(
            kernel_size=(5, 5), sigma=(0.1, 2.0)
        ),  # Random Gaussian blur
        transforms.RandomPerspective(
            distortion_scale=0.5, p=0.5, interpolation=3
        ),  # Random perspective transformation
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Load Data
def load_data(data_dir, batch_size=32):  # Increase batch size to avoid batch norm issue
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_dataset = ImageFolder(train_dir, transform=train_transforms)
    val_dataset = ImageFolder(val_dir, transform=val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, len(train_dataset.classes)


# Create Model
def create_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze all layers except the last block
    for param in list(model.parameters())[:-10]:
        param.requires_grad = False

    # Modify the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model.to(device)


# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-ce_loss)
        return self.alpha * (1 - pt) ** self.gamma * ce_loss


# Training and Validation
def fine_tune_model(
    train_loader, model, val_loader, num_epochs=60, accumulation_steps=2
):
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2
    )

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        optimizer.zero_grad()  # Reset gradients

        for step, (inputs, labels) in enumerate(train_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            # Standard precision (no mixed precision as MPS doesn't support it)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()

            # Gradient accumulation
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_bar.set_postfix(
                {
                    "loss": f"{train_loss / (step + 1):.4f}",
                    "acc": f"{100. * train_correct / train_total:.2f}%",
                }
            )

        scheduler.step()  # Adjust learning rate

        # Validation phase
        model.eval()  # Switch to evaluation mode to avoid batch norm issues
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total
        print(
            f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {100. * val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "new_best_model.pth")
            print("Model saved with improved accuracy!")

    print(f"Best Validation Accuracy: {100. * best_val_acc:.2f}%")


# Main Function
def main():
    data_dir = "/Users/anurag2506/Documents/coat/dataset"
    batch_size = 32  # Increased batch size
    num_epochs = 60

    train_loader, val_loader, num_classes = load_data(data_dir, batch_size=batch_size)
    model = create_model(num_classes)
    fine_tune_model(train_loader, model, val_loader, num_epochs)


if __name__ == "__main__":
    main()
