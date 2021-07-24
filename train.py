from model_dispatcher import MODEL_DISPATCHER
from dataset import KannadaMNISTTrain
from tqdm import tqdm
import torch
import torch.nn as nn


DEVICE = "cpu"
TRAINING_FOLDS_CSV = "../data/train_folds.csv"
IMG_HEIGHT = 28
IMG_WIDTH = 28
EPOCHS = 10

TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 8

MODEL_MEAN = (0.485, 0.456, 0.406)
MODEL_STD = (0.229, 0.224, 0.225)

TRAINING_FOLDS = (0, 1, 2, 3)
VALIDATION_FOLDS = (4,)
BASE_MODEL = "resnet34"


def calculate_loss(output, target):
    loss = nn.CrossEntropyLoss()(output, target)
    return loss


def train(dataset, data_loader, model, optimizer):
    model.train()
    for bi, d in tqdm(
        enumerate(data_loader), total=int(len(dataset) / data_loader.batch_size)
    ):
        image = d["image"]
        label = d["label"]

        image = image.to(DEVICE, dtype=torch.float)
        label = label.to(DEVICE, dtype=torch.long)

        optimizer.zero_grad()
        output = model(image)

        loss = calculate_loss(output, label)
        loss.backward()
        optimizer.step()


def evaluate(dataset, data_loader, model):
    model.eval()
    final_loss = 0
    counter = 0
    for bi, d in tqdm(
        enumerate(data_loader), total=int(len(dataset) / data_loader.batch_size)
    ):
        counter = counter + 1
        image = d["image"]
        label = d["label"]

        image = image.to(DEVICE, dtype=torch.float)
        label = label.to(DEVICE, dtype=torch.long)

        output = model(image)

        loss = calculate_loss(output, label)
        final_loss += loss

    return final_loss / counter  # Mean loss


def main():
    model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True)
    model.to(DEVICE)

    train_dataset = KannadaMNISTTrain(
        folds=TRAINING_FOLDS,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        mean=MODEL_MEAN,
        std=MODEL_STD,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=6,
    )

    # Validation

    valid_dataset = KannadaMNISTTrain(
        folds=VALIDATION_FOLDS,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        mean=MODEL_MEAN,
        std=MODEL_STD,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=6,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.3, verbose=True  # loss,
    )

    # Add early stopping

    for epoch in range(EPOCHS):
        train(train_dataset, train_loader, model, optimizer)
        val_score = evaluate(valid_dataset, valid_loader, model)
        scheduler.step(val_score)
        torch.save(
            model.state_dict(), f"model-{BASE_MODEL}-fold-{VALIDATION_FOLDS[0]}.bin"
        )


if __name__ == "__main__":
    main()
