import pretrainedmodels
import glob
import torch
import albumentations
import joblib
import pandas as pd
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from PIL import Image
from torch.nn import functional as F

TEST_BATCH_SIZE = 32
MODEL_MEAN = (0.485, 0.456, 0.406)
MODEL_STD = (0.229, 0.224, 0.225)
IMG_HEIGHT = 28
IMG_WIDTH = 28
DEVICE = "cpu"  # We are deploying on Heroku CPU


def transform(image):
    perform_transformation = albumentations.Compose(
        [
            albumentations.Resize(IMG_HEIGHT, IMG_WIDTH, always_apply=True),
            albumentations.Normalize(MODEL_MEAN, MODEL_STD, always_apply=True),
        ]
    )
    return perform_transformation(image=image)["image"]


def predict(image):
    image = image.unsqueeze(0)
    image = image.to(DEVICE, dtype=torch.float)
    out = model(image)
    digit = np.argmax(out.cpu().detach().numpy(), axis=1)

    return digit


class resnet34(nn.Module):
    def __init__(self, pretrained):
        super(resnet34, self).__init__()
        if pretrained:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained=None)

        self.out = nn.Linear(512, 10)

    def forward(self, x):  # Takes a batch
        bs, channels, height, width = x.shape
        x = self.model.features(x)  # function of pretrainedmodels

        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        out = self.out(x)

        return out


model = resnet34(pretrained=False)
model = model.to(DEVICE)

model.load_state_dict(
    torch.load(
        "./static/model-resnet34-fold-4-epoch-7.bin",
        map_location=torch.device("cpu"),
    )
)

model.eval()
