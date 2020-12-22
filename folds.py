import pandas as pd
import argparse
from sklearn.model_selection import StratifiedKFold


def create_folds(path):
    df = pd.read_csv(path)
    # create a column for kfold
    df.loc[:, "kfold"] = -1
    labels = df["label"].values
    df["image_id"] = [f"{label}_{i}" for i, label in enumerate(labels)]

    # shuffle the dataframe

    df = df.sample(frac=1).reset_index(drop=True)

    X = df.drop("label", axis=1)
    X = X.values
    y = df["label"].values

    skf = StratifiedKFold(n_splits=5)

    for fold, (train, val) in enumerate(skf.split(X, y)):
        print(f"Train: {train.shape} Val: {val.shape}")
        df.loc[val, "kfold"] = fold

    print(df.kfold.value_counts())
    df.to_csv("../data/train_folds.csv", columns=df.columns, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to Train CSV")
    args = parser.parse_args()
    create_folds(args.path)