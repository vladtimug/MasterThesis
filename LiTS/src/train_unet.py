import wandb
import torch
from tqdm import tqdm
from losses import CrossEntropy
from dataset import LiTSDataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from engine import train_batch, validate_batch

# Logging Infrastructure
run = wandb.init(
    project="LiTS_UNet_e1",
    save_code=True,
    tags=["baseline"]
)

wandb.config = {
    # Training params
    "training_batch_size": 8,
    "training_epochs": 40,
    
    # Validation params
    "validation_batch_size": 4,

    # Model params
    "encoder_name": "resnet34",
    "encoder_depth": 5,
    "encoder_weights": "imagenet",
    "decoder_use_batchnorm": True,
    "decoder_channels": [256, 128, 64, 32, 16],
    "activation": None,
    "learning_rate": 1e-3,

    # Hardware params
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
    batch_size=wandb.config["training_batch_size"],
    shuffle=True,
    collate_fn=train_dataset.collate_fn
)

validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=wandb.config["validation_batch_size"],
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
        encoder_name=wandb.config["encoder_name"],
        encoder_depth=wandb.config["encoder_depth"],
        encoder_weights=wandb.config["encoder_weights"],
        decoder_use_batchnorm=wandb.config["decoder_use_batchnorm"],
        decoder_channels=wandb.config["decoder_channels"],
        decoder_attention_type=None,
        in_channels=1,
        classes=3,
        activation=wandb.config["activation"],
        aux_params=None
    )
model.to(wandb.config["device"])
criterion = CrossEntropy
optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config["learning_rate"])

# Training Loop

max_score = 0

for epoch in range(wandb.config["training_epochs"]):
    for _, data in tqdm(enumerate(train_dataloader), total = len(train_dataloader)):
        train_loss, train_dsc, train_iou = train_batch(model, data, optimizer, criterion)
    
    for _, data in tqdm(enumerate(validation_dataloader), total = len(validation_dataloader)):
        validation_loss, validation_dsc, validation_iou = validate_batch(model, data, criterion)
        if validation_iou > max_score:
            max_score = validation_iou
            torch.save(model.state_dict(), f"{wandb.run.name}_best_iou_{max_score:.4f}.pth")

    wandb.log(
        {
            'train_loss': train_loss,
            'train_background_dice_score': train_dsc[0],
            'train__liver_dice_score': train_dsc[1],
            'train__tumor_dice_score': train_dsc[2],
            'train_IoU': train_iou,
            'val_loss': validation_loss,
            'val_background_dice_score': validation_dsc[0],
            'val_liver_dice_score': validation_dsc[1],
            'val_tumor_dice_score': validation_dsc[2],
            'val_IoU': validation_iou
        }
    )
