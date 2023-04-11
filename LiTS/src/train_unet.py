import wandb
import torch
from tqdm import tqdm
from losses import CrossEntropy
from dataset import LiTSDataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from engine import Engine

# Logging Infrastructure
run = wandb.init(
    project="LiTS_UNet_e1",
    save_code=True,
    tags=["baseline"]
)

wandb.config = {
    "batch_size": 8,
    "training_epochs":20,
    "learning_rate": 1e-3,
    "dataset_normalization": "ImageNet",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Data Preparation
train_dataset = LiTSDataset(
    rootDataDirectory="./data/",
    datasetSplit="train",
    device=wandb.config["device"]
)

validation_dataset = LiTSDataset(
    rootDataDirectory="./data/",
    datasetSplit="validation",
    device=wandb.config["device"]
)

# test_dataset = LiTSDataset(
#     rootDataDirectory="../data/",
#     datasetSplit="test"
# )

train_dataloader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=train_dataset.collate_fn
)

validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=validation_dataset.collate_fn
)

# test_dataloader = DataLoader(
#     test_dataset,
#     batch_size=1,
#     shuffle=False,
#     collate_fn=test_dataset.collate_fn
# )

# Model Preparation
model = smp.Unet(
        encoder_name="resnet34",
        encoder_depth=5,
        encoder_weights="imagenet",
        decoder_use_batchnorm=True,
        decoder_channels=[256, 128, 64, 32, 16],
        decoder_attention_type=None,
        in_channels=1,
        classes=3,
        activation=None,
        aux_params=None
    )
model.to(wandb.config["device"])
criterion = CrossEntropy
optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config["learning_rate"])

# Training Loop
for epoch in range(wandb.config["training_epochs"]):
    for _, data in tqdm(enumerate(train_dataloader), total = len(train_dataloader)):
        train_loss, train_accuracy = Engine.train_batch(model, data, optimizer, criterion)
    
    for _, data in tqdm(enumerate(validation_dataloader), total = len(validation_dataloader)):
        validation_loss, validation_acccuracy = Engine.validate_bacth(model, data, criterion)

    wandb.log(
        {
            'train_loss': train_loss,
            'train_acc': train_accuracy,
            'val_loss': validation_loss,
            'val_acc': validation_acccuracy
        }
    )

torch.save(model.state_dict(), f"{wandb.run.name}.pth")
