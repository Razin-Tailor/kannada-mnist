import albumentations
import joblib
import torch

import pandas as pd
import numpy as np

from PIL import Image


class KannadaMNISTTrain:
    def __init__(self, folds, img_width, img_height, mean, std):
        df = pd.read_csv("../data/train_folds.csv")
        df = df[df.kfold.isin(folds)].reset_index(drop=True)
        self.image_ids = df.image_id.values
        self.label = df.label.values

        # Augmentations

        if len(folds) == 1:  # Validation phase

            self.augmentation = albumentations.Compose(
                [
                    albumentations.Resize(img_height, img_width, always_apply=True),
                    albumentations.Normalize(mean, std, always_apply=True),
                ]
            )

        self.augment = albumentations.Compose(
            [
                albumentations.Resize(img_height, img_width, always_apply=True),
                albumentations.Normalize(mean, std, always_apply=True),
                albumentations.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=5, p=0.9
                ),
            ]
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        image = joblib.load(f"../data/image_pickles/{self.image_ids[item]}.pkl")
        image = image.reshape(28, 28).astype(float)
        image = Image.fromarray(image).convert("RGB")  # WHY?
        # Because all the models that we would try:
        # maybe from torchvision or pretrainedmodels they all work on rgb.
        # So we don't want to spend time on making them work for single channel only

        image = self.augment(image=np.array(image))["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        # Take a look at torchvision models to know why such dtype
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "label": torch.tensor(self.label[item], dtype=torch.long),
        }
