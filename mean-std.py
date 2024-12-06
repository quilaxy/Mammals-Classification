import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Dataset transform
transform = transforms.ToTensor()

# Load dataset
dataset = datasets.ImageFolder(root='dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# Initialize sums and squared sums
mean = torch.zeros(3)
std = torch.zeros(3)
total_pixels = 0

for images, _ in dataloader:
    # Reshape to (batch_size, channels, -1) and calculate stats
    images = images.view(images.size(0), images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    total_pixels += images.size(0)

mean /= total_pixels
std /= total_pixels

print(f"Mean: {mean}")
print(f"Std: {std}")
