import pandas as pd
import cv2
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CovidDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_map = {'COVID-19': 0, 'Bacterial': 1, 'Viral': 2, 'Other': 3}
        self.metadata['label_idx'] = self.metadata['simple_label'].map(self.label_map)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = self.metadata['filename'].iloc[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Cannot load {img_path}, skipping")
            return self.__getitem__((idx + 1) % len(self))  # 跳到下一個
        image = image.astype('float32') / 255.0
        image = cv2.resize(image, (224, 224))
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        label = self.metadata['label_idx'].iloc[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == "__main__":
    data_dir = '/uufs/chpc.utah.edu/common/home/u1527533/covid_dataset'
    csv_file = os.path.join(data_dir, 'metadata_filtered.csv')
    img_dir = os.path.join(data_dir, 'images')
    transform = transforms.Compose([transforms.Normalize(mean=[0.5], std=[0.5])])
    dataset = CovidDataset(csv_file, img_dir, transform)
    print(f"Dataset size: {len(dataset)}")
    image, label = dataset[0]
    print(f"Sample image shape: {image.shape}, Label: {label}")
