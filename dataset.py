import glob
import os

from PIL import Image
from torch.utils.data import Dataset


class AnimalsDataset(Dataset):
    def __init__(self, path, transform=None):
        self.classes = os.listdir(path)
        self.path = [classN for classN in self.classes]
        self.file_list = [glob.glob(f'{x}/*') for x in self.path]
        self.transform = transform

        pictures = []

        for i, classN in enumerate(self.classes):
            for fileName in self.file_list[i]:
                pictures.append([i, classN, fileName])
        self.file_list = pictures
        pictures = None

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        class_category = self.file_list[index][0]
        file_name = self.file_list[index][2]

        img = Image.open(file_name)

        if self.transform:
            img = self.transform(img)

        return img.view(-1), class_category



