import numpy as np
import pandas as pd
import os, argparse, tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from glob import glob
from model_training.preprocessing_utils import normalize, reorient_to_match_training
from model_training import constants

def parse_script_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_data_dir_root_path",
        type=str, help="Path to the root directory of the test dataset",
        default="/home/tvlad/Downloads/test_dataset_3DIRCADB/Test_Data_3Dircadb1/"
    )
    
    parser.add_argument(
        "--inference_dir_root_path",
        type=str, help="Path to the root directory of the test dataset inference results",
        default="/home/tvlad/Projects/MasterThesis/LiTS/experiments_data/set_6_1/experiment_2/test_results_3DIRCADB_Positive/"
    )

    parser.add_argument(
        "--output_dir_path",
        type=str, help="Path to the root directory where results will be stored",
        default="/home/tvlad/Projects/MasterThesis/LiTS/experiments_data/set_6_1/experiment_2/test_results_3DIRCADB_Positive_viz/"
    )

    return parser.parse_args()

def plot_mask_vs_prediction(scan_slice, mask, prediction, save_path, figure_title, save_fig=True):
    if np.min(scan_slice) < 0.:
        multi_channel_slice = np.stack(3 * [(scan_slice * constants.LITS_DATASET_STD) + constants.LITS_DATASET_MEAN], axis=2)
    else:
        multi_channel_slice = np.stack(3 * [scan_slice], axis=2)

    # true positive predictions = green
    intersection = mask * prediction
    multi_channel_slice[np.nonzero(intersection)] = (0.0, 1.0, 0.0)
    
    # false positive predictions = red
    multi_channel_slice[np.nonzero((mask == 0) * (prediction == 1))] = (1.0, 0.0, 0.0)

    # false negative prediction = blue
    multi_channel_slice[np.nonzero((mask == 1) * (prediction == 0))] = (0.0, 1.0, 1.0)

    plt.tight_layout()
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs[0, 0].imshow(scan_slice, cmap="gray")
    axs[0, 0].set_title("Scan Slice")

    axs[0, 1].imshow(mask, cmap="gray")
    axs[0, 1].set_title("Slice Mask")

    axs[1, 0].imshow(prediction, cmap="gray")
    axs[1, 0].set_title("Slice Prediction")

    axs[1, 1].imshow(multi_channel_slice)
    axs[1,1].set_title("Mask vs Prediction")

    plt.suptitle(figure_title)
    plt.legend(handles=[
        mpatches.Patch(color="lawngreen", label="True Positives"),
        mpatches.Patch(color="red", label="False Positives"),
        mpatches.Patch(color="cyan", label="False Negatives")
        ])
    
    if save_fig:
        plt.savefig(save_path)
        plt.close()
    else:
        return fig

if __name__ == "__main__":
    script_args = parse_script_arguments()
    
    if not os.path.exists(script_args.output_dir_path):
        os.mkdir(script_args.output_dir_path)

    MASKS_FILENAME = "2D_LesionMasks_Positive.csv"
    VOLUMES_FILENAME = "2D_Volumes_Positive.csv"
    MASKS_DIRNAME = "LesionMasks"
    VOLUMES_DIRNAME = "Volumes"

    volume_masks = pd.read_csv(os.path.join(script_args.test_data_dir_root_path, MASKS_FILENAME), header=0)
    volume_slices = pd.read_csv(os.path.join(script_args.test_data_dir_root_path, VOLUMES_FILENAME), header=0)
    
    available_volumes = pd.unique(volume_masks["Volume"])
    for volume_name in tqdm.tqdm(available_volumes, total=len(available_volumes)):
        # Create output directory
        output_volume_dir_path = os.path.join(script_args.output_dir_path, volume_name)
        if not os.path.exists(output_volume_dir_path):
            os.mkdir(output_volume_dir_path)

        # Set path to inference results directory
        inference_dirpath = os.path.join(script_args.inference_dir_root_path, volume_name)
        filtered_predictions = sorted(glob(inference_dirpath + "/*.npy"), key=lambda elem: int(elem.split("/")[-1].split("-")[-1].split(".")[0]))

        # Extract all relevant paths
        filtered_masks = volume_masks[volume_masks["Volume"] == volume_name]
        filtered_slices = volume_slices[volume_slices["Volume"] == volume_name]

        # Sanity check for equivalent number of entries
        if not(len(os.listdir(inference_dirpath)) == len(filtered_masks) == len(filtered_slices)):
            raise Exception("Found invalid number of entries among the inference directory content, the number of volume masks and the number of volume slices. They must be equal.")

        # create inference visualization for each slice in volume
        for slice_path, mask_path, prediction_path in zip(filtered_slices["Slice Path"], filtered_masks["Slice Path"], filtered_predictions):
            scan_slice = normalize(np.load(slice_path), zero_center=False, unit_variance=False)
            slice_mask = np.load(mask_path)
            res = np.load(prediction_path)
            slice_prediction = reorient_to_match_training(res)

            slice_name = slice_path.split("/")[-1].split(".")[0]
            plot_path = os.path.join(output_volume_dir_path, slice_name)
            figure_title = volume_name + " " + slice_name
            plot_mask_vs_prediction(scan_slice, slice_mask, slice_prediction, plot_path, figure_title)
            