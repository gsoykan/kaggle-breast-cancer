import os
from typing import Optional

import cv2
import pandas as pd
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import albumentations as A


class BreastDataset(Dataset):
    def __init__(self,
                 csv_file: Optional[str],
                 img_dir: str,
                 transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.is_labeled = csv_file is not None
        if self.is_labeled:
            labeled_csv = pd.read_csv(csv_file)
            self.dataset = [(item['id'], item['prediction']) for item in labeled_csv.to_dict('records')]
        else:
            items = os.listdir(img_dir)
            self.dataset = list(map(lambda x: (x, None), items))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_id, pred = self.dataset[idx]
        img_id_int = int(img_id.split('.')[0])
        img_name = os.path.join(self.img_dir,
                                img_id)

        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        if self.is_labeled:
            return {'image': image, 'pred': pred, 'img_id': img_id_int}
        else:
            return {'image': image, 'img_id': img_id_int}


if __name__ == '__main__':
    csv_path = "/home/gsoykan/Desktop/dev/kaggle-breast-cancer/data/inzva-ml-bootcamp-class-2-kaggle-challenge/train.csv"
    img_dir = "/home/gsoykan/Desktop/dev/kaggle-breast-cancer/data/inzva-ml-bootcamp-class-2-kaggle-challenge/train/train"

    transformations = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])  # training için rotate, scale, pad from sides to not distort the image...

    dataset = BreastDataset(csv_path, img_dir, transform=transformations)
    print(dataset)
