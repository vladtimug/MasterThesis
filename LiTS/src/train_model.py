import os
import wandb
import torch
import random
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import trange, tqdm
from models.classic_unet import UNet
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from model_training.metrics import Metrics
from model_training import constants, engine
from model_training.dataset import LiTSDataset
from model_training.training_config import lesion_config as lesion_training_config
from model_training.losses import MultiClassPixelWiseCrossEntropy, MultiClassCombined
from plot_inference_results import plot_mask_vs_prediction
import matplotlib.pyplot as plt

def Generate_Required_Datasets(config):
    rng = np.random.RandomState(config['seed'])
    vol_info = {}
    vol_info['volume_slice_info'] = pd.read_csv(constants.ROOT_PREPROCESSED_TRAINING_DATA_PATH + '/Assign_2D_Volumes.csv',     header=0)
    vol_info['target_mask_info']  = pd.read_csv(constants.ROOT_PREPROCESSED_TRAINING_DATA_PATH + '/Assign_2D_LesionMasks.csv', header=0) if config['data'] == 'lesion' else pd.read_csv(constants.ROOT_PREPROCESSED_TRAINING_DATA_PATH+'/Assign_2D_LiverMasks.csv', header=0)

    if config['use_weightmaps']:  vol_info['weight_mask_info']  = pd.read_csv(constants.ROOT_PREPROCESSED_TRAINING_DATA_PATH+'/Assign_2D_LesionWmaps.csv', header=0) if config['data'] == 'lesion' else pd.read_csv(constants.ROOT_PREPROCESSED_TRAINING_DATA_PATH+'/Assign_2D_LiverWmaps.csv', header=0)

    available_volumes = sorted(list(set(np.array(vol_info['volume_slice_info']['Volume']))), key=lambda x: int(x.split('-')[-1]))
    rng.shuffle(available_volumes)

    percentage_data_len = int(len(available_volumes)*config['perc_data'])
    train_val_split     = int(percentage_data_len*config['train_val_split'])
    training_volumes    = available_volumes[:percentage_data_len][:train_val_split]
    validation_volumes  = available_volumes[:percentage_data_len][train_val_split:]

    training_dataset   = LiTSDataset(vol_info, training_volumes, config)
    validation_dataset = LiTSDataset(vol_info, validation_volumes, config, is_validation=True)
    return training_dataset, validation_dataset

def Generate_Validation_Predictions_Comparison_Table(model, dataloader, run_config):
    # Log predictions table for samples in the validation split
    table = wandb.Table(columns=['Volume', 'Slice', 'TumorMask', 'TumorPrediction', 'DiceScore', 'IoUScore', 'PredictionToMaskOverlap'], allow_mixed_types = True)
    _ = model.eval()

    for idx, file_dict in tqdm(enumerate(dataloader), total = len(dataloader)):                
        # Handle model input   
        validation_slice = file_dict["input_images"]
        
        model_input = validation_slice.type(torch.FloatTensor).to(run_config["device"])
        
        # Compute model output
        model_output = model(model_input).data.cpu().numpy()

        # Prepare Prediction Mask for Visualization
        if run_config["training_config"]["num_out_classes"] != 1:
            prediction_mask = np.argmax(model_output, axis=1)[0].astype(np.uint8)
        else:
            prediction_mask = np.argmax(np.round(model_output))[0].astype(np.uint8)

        # Prepare Scan Slice for Visualization
        validation_slice = validation_slice.data.cpu().numpy()[0, 0]

        # Prepare Annotated Mask for Visualization
        validation_mask = validation_mask  = file_dict["targets"][0, 0].numpy().astype(np.uint8)

        # Log Results Table
        validation_sample_path = file_dict["internal_slice_name"][0]
        slice_name = validation_sample_path.split("/")[-1]
        volume_name = validation_sample_path.split("/")[-2]
        mask_flag = True if np.count_nonzero(validation_mask) != 0 else False
        prediction_flag = True if np.count_nonzero(prediction_mask) != 0 else False
        dice_score = Metrics.Dice(prediction_mask, validation_mask)
        iou_score = Metrics.IoU(prediction_mask, validation_mask)
        comparison_figure = plot_mask_vs_prediction(validation_slice, validation_mask, prediction_mask, "", figure_title=volume_name + " " + slice_name, save_fig=False)
        log_image = wandb.Image(comparison_figure)
        plt.close(comparison_figure)
        
        table.add_data(
            volume_name,
            slice_name,
            mask_flag,
            prediction_flag,
            dice_score,
            iou_score,
            log_image
        )

    wandb.log({"Best Model - Validation Predictions": table})

if __name__ == "__main__":
    # Logging Infrastructure
    preprocessing_tag = "polar" if lesion_training_config["polar_training"] else "carthesian"
    complexity_tag = None
    if lesion_training_config["lr"] == 1e-3 and lesion_training_config["l2_reg"] == 0 and lesion_training_config["gamma"] == 1:
        complexity_tag = "baseline"
    elif lesion_training_config["lr"] == 3e-5 and lesion_training_config["l2_reg"] == 1e-4 and lesion_training_config["gamma"] == 0.1:
        complexity_tag = "optimized"
    else:
        raise Exception("Unexpected training config")
    
    run = wandb.init(
        project="LiTS_2D_UNet_e1",
        tags=[lesion_training_config["model"], preprocessing_tag, complexity_tag],
    )
    
    # Configs
    wandb.config = {
        # Training Configuration
        "training_config": lesion_training_config,

        # Hardware params
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    # Reproducibility
    torch.manual_seed(wandb.config["training_config"]["seed"])
    torch.cuda.manual_seed(wandb.config["training_config"]["seed"])
    np.random.seed(wandb.config["training_config"]["seed"])
    random.seed(wandb.config["training_config"]["seed"])
    torch.backends.cudnn.deterministic = True

    pkl.dump(wandb.config, open(os.path.join(wandb.run.dir, "experiment_config.pkl"), "wb"))

    # GPU Setup
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(wandb.config["training_config"]["gpu"])

    # Data Preparation
    train_dataset, validation_dataset = Generate_Required_Datasets(wandb.config["training_config"])

    train_dataloader = DataLoader(
        train_dataset,
        num_workers=wandb.config["training_config"]["num_workers"],
        batch_size=wandb.config["training_config"]["batch_size"],
        pin_memory=False,
        shuffle=True
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        num_workers=0,
        batch_size=1,
        shuffle=False
    )

    # Loss Setup
    if wandb.config["training_config"]["loss_func"] == "multiclass_pwce":
        loss = MultiClassPixelWiseCrossEntropy(config=wandb.config)
    elif wandb.config["training_config"]["loss_func"] == "multiclass_combined":
        loss = MultiClassCombined(config=wandb.config)
    else:
        raise NotImplementedError
    
    # Model Setup
    if wandb.config["training_config"]["model"] == "classic_unet":
        model = UNet(
            in_channels=1,
            out_channels=2
        )
    elif wandb.config["training_config"]["model"] == "unet_plus_plus":
        model = smp.UnetPlusPlus(
            in_channels=1,
            classes=2,
            encoder_weights=None,
            activation='sigmoid'
        )
    elif wandb.config["training_config"]["model"] == "deeplab":
        model = smp.DeepLabV3Plus(
            in_channels=1,
            classes=2,
            encoder_weights=None,
            activation="sigmoid"
        )
    model.to(wandb.config["device"])
    
    # Optimizer Setup
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=wandb.config["training_config"]["lr"],
        weight_decay=wandb.config["training_config"]["l2_reg"]
    )

    # Learning Rate Scheduler Setup
    if isinstance(wandb.config["training_config"]["step_size"], list):
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=wandb.config["training_config"]["step_size"],
            gamma=wandb.config["training_config"]["gamma"]
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=wandb.config["training_config"]["step_size"],
            gamma=wandb.config["training_config"]["gamma"]
        )

    # Model Metrics
    logging_keys = ["train_dice", "train_iou", "train_precision", "train_accuracy", "train_recall", "train_specificity", "train_auc_score", "train_loss",
                    "val_dice", "val_iou", "val_precision", "val_accuracy", "val_recall", "val_specificity", "val_auc_score", "val_loss"]
    metrics = {key:[] for key in logging_keys}
    metrics['best_val_dice'] = 0

    # Training Loop Setup
    training_epochs = trange(wandb.config["training_config"]["num_epochs"], position=1)

    for epoch in training_epochs:
        
        # Training
        training_epochs.set_description(f"Training Epoch {epoch} [learning rate = {scheduler.get_last_lr()}]")
        engine.model_trainer(
            model_setup=[model, optimizer],
            data_loader=train_dataloader,
            loss_func=loss,
            device=wandb.config["device"],
            metrics_idx=wandb.config["training_config"]["verbose_idx"],
            metrics=metrics,
            epoch=epoch
        )
        torch.cuda.empty_cache()

        # Validation
        training_epochs.set_description(f"Validating Epoch {epoch}")
        engine.model_validator(
            model=model,
            data_loader=validation_dataloader,
            loss_func=loss,
            device=wandb.config["device"],
            num_classes=wandb.config["training_config"]["num_out_classes"],
            metrics=metrics,
            epoch=epoch
        )
        torch.cuda.empty_cache()

        # Save Training/Best Validation Checkpoint
        training_model_checkpoint_parameters = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "training_loss": metrics["train_loss"][-1]
        }
        torch.save(
                obj=training_model_checkpoint_parameters,
                f=os.path.join(wandb.run.dir, "latest_epoch_model.pth")
            )
        
        if metrics["val_dice"][-1] > metrics["best_val_dice"]:
            best_validation_model_checkpoint_parameters = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "training_loss": metrics["train_loss"][-1],
                "validation_loss": metrics["val_loss"][-1],
                "training_dice_score": metrics["train_dice"][-1],
                "validation_dice_score": metrics["val_dice"][-1],
                "training_iou_score": metrics["train_iou"][-1],
                "validation_iou_score": metrics["val_iou"][-1],
                "training_precision": metrics["train_precision"][-1],
                "validation_precision": metrics["val_precision"][-1],
                "training_accuracy": metrics["train_accuracy"][-1],
                "validation_accuracy": metrics["val_accuracy"][-1],
                "training_recall": metrics["train_recall"][-1],
                "validation_recall": metrics["val_recall"][-1],
                "training_specificity": metrics["train_specificity"][-1],
                "validation_specificity": metrics["val_specificity"][-1],
                "training_auc_score": metrics["train_auc_score"][-1],
                "validation_auc_score": metrics["val_auc_score"][-1]
            }

            torch.save(
                obj=best_validation_model_checkpoint_parameters,
                f=os.path.join(wandb.run.dir, "best_val_dice.pth")
            )

            metrics["best_val_dice"] = metrics["val_dice"][-1]

        # Log Data
        wandb.log(
            {
                "train_loss": metrics["train_loss"][-1],
                "train_dice": metrics["train_dice"][-1],
                "train_iou": metrics["train_iou"][-1],
                "train_precision": metrics["train_precision"][-1],
                "train_accuracy": metrics["train_accuracy"][-1],
                "train_recall": metrics["train_recall"][-1],
                "train_specificity": metrics["train_specificity"][-1],
                "train_auc_score": metrics["train_auc_score"][-1],
                "val_loss": metrics["val_loss"][-1],
                "val_dice": metrics["val_dice"][-1],
                "val_iou": metrics["val_iou"][-1],
                "val_precision": metrics["val_precision"][-1],
                "val_accuracy": metrics["val_accuracy"][-1],
                "val_recall": metrics["val_recall"][-1],
                "val_specificity": metrics["val_specificity"][-1],
                "val_auc_score": metrics["val_auc_score"][-1],
            }
        )

        pd.DataFrame(metrics).to_csv(os.path.join(wandb.run.dir, "experiment_history.csv"), index=False)
        
        scheduler.step()

    Generate_Validation_Predictions_Comparison_Table(model=model, dataloader=validation_dataloader, run_config=wandb.config)
