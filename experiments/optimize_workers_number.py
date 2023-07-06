from time import time
import multiprocessing as mp
from dataset import LiTSDataset
from torch.utils.data import DataLoader
from train_unet import Generate_Required_Datasets
from training_configs import liver_config as liver_training_config

# Data Preparation
train_dataset, validation_dataset = Generate_Required_Datasets(liver_training_config)

# train_dataloader = DataLoader(
#     train_dataset,
#     num_workers=wandb.config["training_config"]["num_workers"],
#     batch_size=wandb.config["training_config"]["batch_size"],
#     pin_memory=False,
#     shuffle=True
# )

for num_workers in range(2, mp.cpu_count(), 2):  
    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              num_workers=num_workers,
                              batch_size=2,
                              pin_memory=True)
    start = time()
    for epoch in range(1, 3):
        for i, data in enumerate(train_loader, 0):
            pass
    end = time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))