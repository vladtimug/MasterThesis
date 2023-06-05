import os
import torch
import argparse
from unet_classic import UNet
from unet import Scaffold_UNet
import segmentation_models_pytorch as smp
from model_configs import lesion_config as lesion_model_config
from training_configs import lesion_config as lesion_training_config

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path",
                        type=str,
                        default="./LiTS/experiments_data/set_3/lesion/experiment_7/"
                    )

    script_arguments = parser.parse_args()
    return script_arguments

if __name__ == "__main__":

    # Parse script arguments
    args = parse_arguments()

    # Get model state dict
    state_dict_name = "best_val_dice.pth"
    best_validation_checkpoint = torch.load(os.path.join(args.model_path, state_dict_name))

    # Prepare model prereqs
    input_channels = 1
    input_shape = (512, 512)
    input_data   = torch.randn((1, input_channels, *input_shape)).type(torch.FloatTensor)

    # Instantiate model & set its state dict
    model = UNet(
        in_channels=1,
        out_channels=2
    )
    model.load_state_dict(best_validation_checkpoint["model_state_dict"])

    # Convert model to onnx format
    model_name = "best_val_dice.onnx"
    torch.onnx.export(
        model,
        input_data,
        os.path.join(args.model_path, model_name),
        input_names=["input"],
        output_names=["output"]
    )