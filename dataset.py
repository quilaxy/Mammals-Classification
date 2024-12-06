import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

class SimpleTorchDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, aug: list = []) -> None:
        self.dataset: list[tuple[str, np.ndarray]] = []
        self.root_dir = root_dir

        self.__add_dataset__("camel",            [1, 0, 0, 0, 0, 0, 0])
        self.__add_dataset__("dolphin",          [0, 1, 0, 0, 0, 0, 0])
        self.__add_dataset__("koala",            [0, 0, 1, 0, 0, 0, 0])
        self.__add_dataset__("orangutan",        [0, 0, 0, 1, 0, 0, 0])
        self.__add_dataset__("snow_leopard",     [0, 0, 0, 0, 1, 0, 0])
        self.__add_dataset__("water_buffalo",    [0, 0, 0, 0, 0, 1, 0])
        self.__add_dataset__("zebra",            [0, 0, 0, 0, 0, 0, 1])

        post_processing = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4804, 0.4785, 0.4361], std=[0.2149, 0.2122, 0.2146])
        ]

        self.augmentation = transforms.Compose(
            [transforms.Resize((177, 177))] +
            aug +
            post_processing
        )

    def __add_dataset__(self, dir_name: str, class_label: list[int]) -> None:
        full_path = os.path.join(self.root_dir, dir_name)
        if not os.path.exists(full_path):
            print(f"Warning: Directory {full_path} does not exist.")
            return

        label = np.array(class_label)
        for fname in os.listdir(full_path):
            fpath = os.path.join(full_path, fname)
            if fpath.lower().endswith(('jpg', 'png')):
                self.dataset.append((fpath, label))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        fpath, label = self.dataset[index]

        image = Image.open(fpath).convert('RGB')
        image = self.augmentation(image)

        image = (image - image.min()) / (image.max() - image.min())
        label = torch.Tensor(label)

        return image, label


if __name__ == "__main__":
    print("Testing Datasets")

    # Test the single image dataset
    image_dataset = SimpleTorchDataset('./dataset/train')
    print(f"Total samples in image dataset: {len(image_dataset)}")
    image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=4)
    for x, y in image_loader:
        print(x.shape)
        print(y.shape)
        break
