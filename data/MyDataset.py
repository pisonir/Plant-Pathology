from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import torch
import os
import torchvision.transforms.functional as F


class MyDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.img_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_src = os.path.join(self.root_dir, self.img_df.iloc[idx, 0]+'.jpg')
        image = cv2.imread(img_src, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labels = self.img_df.iloc[idx, 1:].values
        labels_idx = np.argmax(labels)
        labels_idx = torch.tensor(labels_idx, dtype=torch.int64)

        if self.transform:
            image = self.transform(F.to_pil_image(image))

        return image, labels_idx


