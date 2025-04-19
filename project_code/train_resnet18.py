import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from covid_dataset import CovidDataset
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
matplotlib.use('Agg')

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_name):
    train_losses, val_losses = [], []
    best_val_f1 = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f"{model_name}_best.pth")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{model_name}_loss_curve.png")
    plt.close()

    cm = confusion_matrix(val_labels, val_preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"{model_name} Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(4)
    plt.xticks(tick_marks, ['COVID-19', 'Bacterial', 'Viral', 'Other'])
    plt.yticks(tick_marks, ['COVID-19', 'Bacterial', 'Viral', 'Other'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], horizontalalignment="center")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{model_name}_confusion_matrix.png")
    plt.close()

    return train_losses, val_losses, val_f1

if __name__ == "__main__":
    data_dir = '/uufs/chpc.utah.edu/common/home/u1527533/covid_dataset'
    csv_file = os.path.join(data_dir, 'metadata_filtered.csv')
    img_dir = os.path.join(data_dir, 'images')
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = CovidDataset(csv_file, img_dir, transform)
    indices = np.arange(len(dataset))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resnet_model = models.resnet18(weights=None)
    resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 4)
    resnet_model = resnet_model.to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0, 5.0, 2.0]).to(device))
    optimizer = optim.Adam(resnet_model.parameters(), lr=0.001)
    train_model(resnet_model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device, model_name="resnet18")
