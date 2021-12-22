import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


# dataの数が少ないのでdataをappendで複数回結合したあと、trans_ransomで写真をランダムに変形
class Mydataset(Dataset):
    def __init__(self, data):
        self.data = data

        img_list = []
        self.label_list = []
        for i in range(20):
            for img, label in data:
                img_list.append(img)
                self.label_list.append(label)
        self.img_list = torch.stack(img_list)
        self.data_list = []
        for i in range(len(self.label_list)):
            self.data_list.append((self.img_list[i], self.label_list[i]))

        self.trans_random = transforms.Compose([
            transforms.RandomAffine(degrees=[-10, 10], translate=(0.1, 0.1), scale=(0.5, 1.5)),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # trans_randomの引数にはImageが入るためImage.fromarrayでdataのtensorを一旦変換

        img = self.data_list[index][0][0].numpy()
        img = self.trans_random(Image.fromarray(np.uint8(img * 255)))
        label = self.data_list[index][1]

        return img, label
