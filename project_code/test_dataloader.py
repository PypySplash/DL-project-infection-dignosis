from torch.utils.data import DataLoader
from covid_dataset import CovidDataset
from torchvision import transforms
import os

data_dir = '/uufs/chpc.utah.edu/common/home/u1527533/covid_dataset'
csv_file = os.path.join(data_dir, 'metadata_simplified.csv')
img_dir = os.path.join(data_dir, 'images')
transform = transforms.Compose([transforms.Normalize(mean=[0.5], std=[0.5])])

dataset = CovidDataset(csv_file, img_dir, transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

for images, labels in dataloader:
    print(f"Batch images shape: {images.shape}, Labels: {labels}")
    break
