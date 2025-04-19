import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from covid_dataset import CovidDataset
from torchvision import transforms
import os

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# 測試模型
if __name__ == "__main__":
    data_dir = '/uufs/chpc.utah.edu/common/home/u1527533/covid_dataset'
    csv_file = os.path.join(data_dir, 'metadata_simplified.csv')
    img_dir = os.path.join(data_dir, 'images')
    transform = transforms.Compose([transforms.Normalize(mean=[0.5], std=[0.5])])

    dataset = CovidDataset(csv_file, img_dir, transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    model = SimpleCNN(num_classes=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 測試一個批次
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        print(f"Batch loss: {loss.item()}, Outputs shape: {outputs.shape}")
        break
