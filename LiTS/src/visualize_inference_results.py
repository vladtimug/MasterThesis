import os
import cv2 as cv
import streamlit as st

st.set_page_config(layout="wide")
st.title("Liver Tumor Segmentation Results Comparison")

available_volumes = [
    "volume-1", "volume-2", "volume-3", "volume-3", "volume-4", "volume-6", "volume-8", "volume-9",
    "volume-10", "volume-12", "volume-13", "volume-15", "volume-16", "volume-17", "volume-19"
    ]

available_models = {
    "UNet": {
        "results_root_path":"/home/tvlad/Projects/MasterThesis/LiTS/experiments_data/set_3/lesion/",
        "carthesian_model_results_dirname": "experiment_7",
        "polar_model_results_dirname": "experiment_6"
        },
    "UNet++": {
        "results_root_path": "/home/tvlad/Projects/MasterThesis/LiTS/experiments_data/set_4/lesion/",
        "carthesian_model_results_dirname": "experiment_8",
        "polar_model_results_dirname": "experiment_9"
        },
    "DeepLabV3+": {
        "results_root_path": "/home/tvlad/Projects/MasterThesis/LiTS/experiments_data/set_6/lesion/",
        "carthesian_model_results_dirname": "experiment_12",
        "polar_model_results_dirname": "experiment_13"
        }
    }

TEST_RESULTS_DIRNAME = "test_results_viz"

with st.expander("Filters"):
    model_arch = st.selectbox("Model Architecture", list(available_models.keys()))
    test_volume = st.selectbox("Volume", available_volumes)
    
model_dict = available_models[model_arch]

carthesian_results_root_dirpath = os.path.join(model_dict["results_root_path"], model_dict["carthesian_model_results_dirname"], TEST_RESULTS_DIRNAME, test_volume)
polar_results_root_path = os.path.join(model_dict["results_root_path"], model_dict["polar_model_results_dirname"], TEST_RESULTS_DIRNAME, test_volume)

if len(os.listdir(carthesian_results_root_dirpath)) != len(os.listdir(polar_results_root_path)):
    raise Exception("Inconsistent number of results")

first_slice_idx, last_slice_idx = st.select_slider(
    "Select Volume Scan Slice Range",
    options=list(range(len(os.listdir(carthesian_results_root_dirpath)))),
    value=(0, 5)
)

for slice_idx in range(first_slice_idx, last_slice_idx):
    slice_name = f"slice-{slice_idx}.png"

    carthesian_res = cv.imread(os.path.join(carthesian_results_root_dirpath, slice_name))
    polar_res = cv.imread(os.path.join(polar_results_root_path, slice_name))

    st.image(
        image=[carthesian_res, polar_res],
        caption=["Carthesian Model", "Polar Model"],
        channels="BGR",
        width=860
        )
