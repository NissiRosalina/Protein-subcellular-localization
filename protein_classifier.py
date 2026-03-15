
import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import timm


# Configuration
class Config:
    def __init__(self):
        self.DATA_DIR      = "/kaggle/input/competitions/hpa-single-cell-image-classification"
        self.TRAIN_CSV     = self.DATA_DIR + "/train.csv"
        self.TRAIN_IMG_DIR = self.DATA_DIR + "/train"

        self.MODEL_NAME    = "efficientnet_b0"
        self.NUM_CLASSES   = 19
        self.IMG_SIZE      = 256
        self.BATCH_SIZE    = 32
        self.NUM_EPOCHS    = 5
        self.LEARNING_RATE = 3e-4
        self.WEIGHT_DECAY  = 1e-5
        self.TRAIN_SPLIT   = 0.85

        self.SUBSET_SIZE   = 3000       
        self.DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.NUM_WORKERS   = 2
        self.SEED          = 42

config = Config()

#  Dataset
class HPADataset(Dataset):
    def __init__(self, df, img_dir, img_size=256, is_train=True):
        self.df       = df.reset_index(drop=True)
        self.img_dir  = img_dir
        self.img_size = img_size
        self.is_train = is_train
        self.df["label_list"] = self.df["Label"].apply(
            lambda x: [int(i) for i in str(x).split("|")]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]["ID"]

        red    = np.array(Image.open(f"{self.img_dir}/{img_id}_red.png"))
        green  = np.array(Image.open(f"{self.img_dir}/{img_id}_green.png"))
        blue   = np.array(Image.open(f"{self.img_dir}/{img_id}_blue.png"))
        yellow = np.array(Image.open(f"{self.img_dir}/{img_id}_yellow.png"))

        image = np.stack([red, green, blue, yellow], axis=-1)
        image = Image.fromarray(image.astype(np.uint8))
        image = image.resize((self.img_size, self.img_size))
        image = np.array(image)

        if self.is_train:
            if random.random() > 0.5:
                image = np.fliplr(image)
            if random.random() > 0.5:
                image = np.flipud(image)

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = np.zeros(config.NUM_CLASSES, dtype=np.float32)
        for l in self.df.iloc[idx]["label_list"]:
            label[l] = 1.0
        return image, torch.tensor(label)

#  Model
class ProteinClassifier(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained)

        old_conv = self.backbone.conv_stem
        self.backbone.conv_stem = nn.Conv2d(
            4,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        with torch.no_grad():
            self.backbone.conv_stem.weight[:, :3] = old_conv.weight
            self.backbone.conv_stem.weight[:, 3]  = old_conv.weight.mean(dim=1)

        # Step 2: replace final classifier head
        n_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(n_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

#  Utilities
def calculate_f1(preds, targets, threshold=0.5):
    preds   = (preds > threshold).astype(int)
    targets = targets.astype(int)
    return f1_score(targets, preds, average="samples")

#  Training & Validation Loops
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0

    for images, labels in tqdm(loader):
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion):
    model.eval()
    total_loss  = 0
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            outputs = model(images)
            loss    = criterion(outputs, labels)
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            total_loss += loss.item()

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    f1          = calculate_f1(all_preds, all_targets)

    return total_loss / len(loader), f1

#  Main
def main():
    # Load & optionally subset the data
    df = pd.read_csv(config.TRAIN_CSV)
    if config.SUBSET_SIZE:
        df = df.sample(config.SUBSET_SIZE, random_state=config.SEED)

    train_df, val_df = train_test_split(
        df,
        test_size=1 - config.TRAIN_SPLIT,
        random_state=config.SEED
    )

    # Build datasets and loaders
    train_dataset = HPADataset(train_df, config.TRAIN_IMG_DIR, config.IMG_SIZE, True)
    val_dataset   = HPADataset(val_df,   config.TRAIN_IMG_DIR, config.IMG_SIZE, False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    # model, loss func, optimizer
    model     = ProteinClassifier(config.MODEL_NAME, config.NUM_CLASSES).to(config.DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Training loop
    best_f1 = 0

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")

        train_loss       = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_f1 = validate(model, val_loader, criterion)

        print("Train Loss:", train_loss)
        print("Val Loss  :", val_loss)
        print("Val F1    :", val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "best_model.pth")
            print("  -> Best model saved")

    print("\nTraining complete.")
    print("Best F1:", best_f1)


main()
