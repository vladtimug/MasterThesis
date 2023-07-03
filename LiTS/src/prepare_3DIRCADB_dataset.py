import os
import csv
import shutil
import pydicom
import zipfile
import argparse
import numpy as np
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_2_training_volumes", type=str, default='/home/tvlad/Downloads/original_data_3DIRCADB')
    parser.add_argument("--output_path", type=str, default='/home/tvlad/Downloads/test_data_3DIRCADB')

    return parser.parse_args()

def merge_annotated_volumes(scan_path:str, masks_path:str, save_path:str) -> None:
    """
    Convert multiple annotated volumes into a single volume
    """

    for slice_idx in range(len(os.listdir(scan_path))):
        slice_data = []
        slice_idx_paths = glob(masks_path + "livertumor*/image_" + str(slice_idx))
        for slice_on_idx in slice_idx_paths:
            if "livertumor" in slice_on_idx:
                slice_on_idx_data = pydicom.read_file(slice_on_idx).pixel_array
                slice_data.append(slice_on_idx_data)
        
        if len(slice_data) > 0:
            stacked_slices = np.stack(slice_data, axis=2)
            slice_mask = np.sum(stacked_slices, axis=2)

            # Read corresponding slice scan
            try:
                scan_slice = pydicom.read_file(os.path.join(masks_path, "livertumor", f"image_{slice_idx}"))
            except:
                try:
                    scan_slice = pydicom.read_file(os.path.join(masks_path, "livertumor01", f"image_{slice_idx}"))
                except:
                    try:
                        scan_slice = pydicom.read_file(os.path.join(masks_path, "livertumors", f"image_{slice_idx}"))
                    except:
                        scan_slice = pydicom.read_file(os.path.join(masks_path, "livertumor1", f"image_{slice_idx}"))

            # Create an object with the pixel_array set from slice_mask and the metadata of the scan_slice object
            scan_slice.PixelData = slice_mask.astype(np.uint8)

            if not os.path.exists(save_path):
                os.mkdir(save_path)

            pydicom.filewriter.dcmwrite(os.path.join(save_path, f"image_{slice_idx}"), scan_slice)

if __name__ == "__main__":
    script_args = parse_args()
    SLICE_DIR_NAME = "PATIENT_NPY"
    MASKS_DIR_NAME = "MASKS_NPY"
    
    # Create a single dicom directory containing all the annotations for liver tumor from a single scan
    for idx, scan_dir in enumerate(os.listdir(script_args.path_2_training_volumes)):
        if scan_dir.split(".")[-1] in ["5", "7", "11", "14", "18"]:
            continue

        dicom_scan_path = os.path.join(script_args.path_2_training_volumes, scan_dir, "PATIENT_DICOM")
        dicom_masks_path = os.path.join(script_args.path_2_training_volumes, scan_dir, "MASKS_DICOM")
        print(f"{idx + 1}/{len(os.listdir(script_args.path_2_training_volumes))}  {dicom_scan_path}")

        if not os.path.exists(dicom_scan_path):
            # Unzip scans
            with zipfile.ZipFile(dicom_scan_path + ".zip", 'r') as dicom_zip:
                dicom_zip.extractall(os.path.join(script_args.path_2_training_volumes, scan_dir))
            
            npy_scan_path = os.path.join(script_args.path_2_training_volumes, scan_dir, SLICE_DIR_NAME)
            if not os.path.exists(npy_scan_path):
                os.makedirs(npy_scan_path)

            for slice_name in os.listdir(dicom_scan_path):
                scan_data = pydicom.dcmread(os.path.join(dicom_scan_path, slice_name))
                np.save(os.path.join(npy_scan_path, slice_name), scan_data.pixel_array)

        if not os.path.exists(dicom_masks_path):
            # Unzip annotations
            with zipfile.ZipFile(dicom_masks_path + ".zip", 'r') as dicom_zip:
                dicom_zip.extractall(os.path.join(script_args.path_2_training_volumes, scan_dir))
            
            # Create output dir
            output_masks_dir_path = os.path.join(dicom_masks_path, "liver_tumors")

            # Add all slices for a single index
            merge_annotated_volumes(dicom_scan_path, dicom_masks_path, output_masks_dir_path)
        
            # Convert mask slices to npy files
            masks_scan_path = os.path.join(dicom_masks_path, "liver_tumors")
            
            npy_masks_path = os.path.join(script_args.path_2_training_volumes, scan_dir, MASKS_DIR_NAME)
            if not os.path.exists(npy_masks_path):
                os.makedirs(npy_masks_path)

            for slice_name in os.listdir(masks_scan_path):
                scan_data = pydicom.dcmread(os.path.join(masks_scan_path, slice_name))
                np.save(os.path.join(npy_masks_path, slice_name), scan_data.pixel_array)

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    volume_scans = sorted(glob(f"{script_args.path_2_training_volumes}/3Dircadb*"), key=lambda path: int(path.split(".")[-1]))

    if not os.path.exists(script_args.output_path):
        os.mkdir(script_args.output_path)

    slice_ledger = open(f"{script_args.output_path}/2D_Volumes_Positive.csv", "w", encoding="UTF-8")
    mask_ledger = open(f"{script_args.output_path}/2D_LesionMasks_Positive.csv", "w", encoding="UTF-8")

    slice_ledger_writer = csv.writer(slice_ledger)
    mask_ledger_writer = csv.writer(mask_ledger)

    slice_ledger_writer.writerow(["Volume", "Slice Path"])
    mask_ledger_writer.writerow(["Volume", "Slice Path"])

    for volume_scan_path in volume_scans:
        volume_slices_path = os.path.join(volume_scan_path, SLICE_DIR_NAME)
        volume_masks_path = os.path.join(volume_scan_path, MASKS_DIR_NAME)

        volume_scan_id = f"volume-{volume_scan_path.split('.')[-1]}"

        test_volume_slices_path = os.path.join(script_args.output_path, "Volumes", volume_scan_id)
        test_volume_masks_path = os.path.join(script_args.output_path, "LesionMasks", volume_scan_id)

        if not os.path.exists(test_volume_slices_path):
            os.makedirs(test_volume_slices_path)
        
        if not os.path.exists(test_volume_masks_path):
                os.makedirs(test_volume_masks_path)
        
        slice_paths = sorted(glob(volume_slices_path + "/*"), key=lambda x: int(x.split("/")[-1].split("_")[-1].split(".")[0]))
        masks_paths = sorted(glob(volume_masks_path + "/*"), key=lambda x: int(x.split("/")[-1].split("_")[-1].split(".")[0]))

        for slice_path, mask_path in zip(slice_paths, masks_paths):
            if np.count_nonzero(np.load(mask_path)):
                slice_idx = slice_path.split("/")[-1].split("_")[-1].split(".")[0]
                mask_idx = mask_path.split("/")[-1].split("_")[-1].split(".")[0]

                if slice_idx != mask_idx:
                    raise IndexError(f"Slice index and mask index should match. Slice index: {slice_idx} Mask index: {mask_idx}")
                
                slice_save_path = os.path.join(test_volume_slices_path, f"slice-{slice_idx}.npy")
                mask_save_path = os.path.join(test_volume_masks_path, f"slice-{slice_idx}.npy")
                
                slice_ledger_writer.writerow([volume_scan_id, os.path.abspath(slice_save_path)])
                mask_ledger_writer.writerow([volume_scan_id, os.path.abspath(mask_save_path)])
                
                shutil.copy(slice_path, slice_save_path)
                shutil.copy(mask_path, mask_save_path)