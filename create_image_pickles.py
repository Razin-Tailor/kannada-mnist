"""
Convert the df rows to image pickles
why? because using df rows to read during training will be super slow
and by creating pickles we can upload it to cloud and do processing there too

"""

import pandas as pd
from tqdm import tqdm
import joblib


def create_pickles():
    df = pd.read_csv("../data/train.csv")
    labels = df["label"].values
    df = df.drop("label", axis=1)
    image_array = df.values

    for i, label in tqdm(enumerate(labels), total=len(labels)):
        joblib.dump(image_array[i, :], f"../data/image_pickles/{label}_{i}.pkl")


if __name__ == "__main__":
    create_pickles()
