from torch.utils.data import Dataset
from PIL import Image
import torch


class MyDataset(Dataset):
    def __init__(self,image_path:list,image_cla:list,transform=None):
        self.image_path = image_path
        self.image_cla = image_cla
        self.transform = transform
    def __len__(self):
        return len(self.image_path)
    def __getitem__(self, index):
        image = Image.open(self.image_path[index])
        if image.mode !="RGB":
            raise ValueError("image:{} is not RGB mode!".format(self.image_path[index]))
        label = self.image_cla[index]

        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
