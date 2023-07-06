import os
import cv2 as cv
import streamlit as st

st.set_page_config(layout="wide")
st.title("Liver Tumor Segmentation Results")

ircadb_volumes = [
    "volume-1", "volume-2", "volume-3", "volume-4", "volume-6", "volume-8", "volume-9",
    "volume-10", "volume-12", "volume-13", "volume-15", "volume-16", "volume-17", "volume-19"
    ]

acadtum_volumes = {
    "volume-1":'SZTOJKA_IOSIF_HCC_sg_4_8_SZTOJKA_IOSIF_May_28__2020_',
    "volume-2":'FRATEAN_IOSIF_HCC_sg_4_8_CT_19_03_2021_FRATEAN_IOSIF_Mar_19__2021_',
    "volume-3":'BOGHIU_VENIAMIN_HCC_hipovascular_CT_27_09_2022_BOGHIU___VENIAMIN_STEFAN_Sep_27__2022_',
    "volume-4":'DANCU_IOAN_noduli_regenerare_CT_6_08_2021_DANCU_IOAN_Aug_6__2021_',
    "volume-5":'SZATHMARI_SANDOR_JANOS_Noduli_cirotici_suspecti_HCC_SZATHMARI_SANDOR_Apr_11__2017_',
    "volume-6":'Rus_Aurelia__HCC_incipient_sau_nodul_displazic_sg__2__Rus_Aurelia_Jun_5__2020_',
    "volume-7":'GLIGOR_VIORICA__noduli_displazici_GLIGOR____VIORICA_SILVIA_May_28__2021_',
    "volume-8":'KALANIOS_ANTON_Tulburare_perfuzie__fara_HCC_KALANIOS_ANTON_Jun_11__2021_',
    "volume-9":'VINCZE_CSABA_noduli_tratati_prin_ablatie__fara_recidiva_VINCZE_CSABA_Dec_21__2021_',
    "volume-10":'BRANDAS_VASILE_HCC_difuz_CT_9_07_2021_BRANDAS_VASILE_Jul_9__2021_',
    "volume-11":'LACATUS_MARIN__HCC_difuz_invazie_VP_LACATUS_MARIN_Apr_15__2021_',
    "volume-12":'BORDEIANU_STEFAN_HCC_multicentric_CT_9_03_2021_BORDEIANU_STEFAN_LEONARD_Mar_9__2021_',
    "volume-13":'Hideg_Marin_recidiva_nodul_ablat_sg__8_Hideg_Marin_Sep_15__2020_',
    "volume-14":'Oros_Sabin__Noduli_cirotici_displazici_Oros_Sabin_Nov_23__2020_',
    "volume-15":'FILIP_TEODOR_HCC_CT_13_07_2021_FILIP_TEODOR_Jul_13__2021_',
    "volume-16":'BOTEZATU_ELENA_noduli_cirotici_benigni_CT_1_03_2019_BOTEZATU_ELENA_Mar_1__2019_',
    "volume-17":'MIC_NASTASIA_nodul_displazic_segm_2_MIC_NASTASIA_Jun_28__2020_',
    "volume-18":'TIMAR_IOAN_HCC_sg_5_6_TIMAR_IOAN_Jun_2__2021_',
    "volume-19":'MESZAROS_ZOLTAN__HCC_sg_2_MESZAROS_ZOLTAN_May_24__2021_',
    "volume-20":'TOMA_NELU_colangiocarcinom_periferic_TOMA_NELU_Sep_18__2020_',
    "volume-21":'BURLAN_DOREL_HCC_multicentric_CT_5_02_2020_BURLAN_DOREL_Feb_5__2020_',
    "volume-22":'SZATHMARI_SANDOR_JANOS_Noduli_cirotici_suspecti_HCC_SZATHMARI__SANDOR_JANOS_Mar_15__2021_',
    "volume-23":'PETROVICI_LAZAR_nodul_displazic_PETROVICI_LAZAR_Jan_30__2019_',
    "volume-24":'BORDEIANU_STEFAN_HCC_multicentric_CT_12_09_2022_BORDEIANU_STEFAN_Sep_12__2022_',
    "volume-25":'DEAK_EVA_hemangiom_CT_2_07_2020_DEAK_EVA_Jul_2__2020_',
    "volume-26":'MISCHIE_MONICA__HCC_sg_7__noduli_ablati_in_rest__multicentric_pe_RM_MISCHIE_MONICA_Oct_12__2020_',
    "volume-27":'VACARIU_SILVIU_HCC_sg_5_6_VACARIU_SILVIU_Jan_14__2020_',
    "volume-28":'PUSCAS__IOAN_HCC_sg__4a_si_6__in_evolutie_difuz_in_LD_PUSCAS_IOAN_Jan_11__2022_',
    "volume-29":'KOVACS_IULIU_HCC_multicentric_KOVACS_IULIU_Apr_15__2021_',
    "volume-31":'SOMESAN_NICOLAE_mici_metastaza_de_la_cancer_pancreatic_SOMESAN_NICOLAE_Feb_6__2021_',
    "volume-32":'COCIS_PETRE_HCC_segm_2_CT_26_11_2020_COCIS_PETRE_Nov_26__2020_',
    "volume-33":'TONCA_GHEORGHE_HCC_difuz_LD_TONCA_GHEORGHE_Mar_11__2021_',
    "volume-34":'VIDREAN_CAPUSAN_EMIL_metastaze_VIDREAN_CAPUSAN_EMIL_Mar_10__2020_',
    "volume-35":'TELECAN_TRAIAN_colangiocarcinom_intrahepatic_lob_stg_TELECAN_TRAIAN_Mar_8__2021_',
    "volume-36":'HORVATH_MAGDALENA_HCC_ablat_fara_recidiva_HORVATH_MAGDALENA_Oct_6__2020_',
    "volume-37":'KIS_ARPAD_HCC_incipient_sg__7_KIS_ARPAD_Dec_17__2019_',
    "volume-38":'SOOS_MARTIN_ZOLTAN_HCC_sg_7_spre_8_SOOS_MARTIN_Nov_16__2020_',
    "volume-39":'NEAMTU_DORU_HCC_multicentric_NEAMTU_DORU_Feb_9__2021_',
    "volume-40":'SZASZ_VICTOR_HCC_multicentric_SZASZ_VICTOR_TIBERIU_Dec_14__2020_',
    "volume-41":'Grumeza_Valeria_HCC_sg_V_ablat_ulterior_Grumeza_Valeria_Oct_7__2020_',
    "volume-42":'URSACHE_CONSTANTIN__noduli_displazici_sau_HCC_multicentric_incipient_URSACHE_CONSTANTIN_Oct_20__2021_',
    "volume-43":'Grumeza_Valeria_HCC_sg_V_ablat_ulterior_Grumeza_Valeria_Jan_11__2021_',
    "volume-44":'NEAG_TEODOR_HCC_multicentric_hipovasc_NEAG_TEODOR_Mar_28__2021_',
    "volume-45":'PUSCAS__IOAN_HCC_sg__4a_si_6__in_evolutie_difuz_in_LD_PUSCAS__IOAN_Jan_6__2020_',
    "volume-46":'SZABO__ALEXANDRU_HCC_sg__7_SZABO_ALEXANDRU_Nov_9__2020_',
    "volume-47":'NISTOR_STEFAN__HCC_difuz_NISTOR_STEFAN_Feb_19__2021_',
    "volume-48":'BUSUIOC_VIOREL_colangiocarcinom_CT_20_09_2020_BUSUIOC_VIOREL_Sep_20__2020_',
    "volume-49":'CUPAR_DORINA_noduli_regenerare_CT_28_03_2021_CUPAR_DORINA_May_28__2021_',
    "volume-50":'ALBU_DUMITRU_HCC_multicentric_CT_17_08_2020_ALBU_DUMITRU_Aug_17__2020_'
}

available_datasets = {
    "3DIRCADB": ircadb_volumes,
    # "ACADTUM": acadtum_volumes
}

available_models = {
    "UNet": {
        "results_root_path":"/home/tvlad/Projects/MasterThesis/LiTS/experiments_data/set_3_1/",
        "carthesian_model_results_dirname": "experiment_1",
        "polar_model_results_dirname": "experiment_2"
        },
    "UNet++": {
        "results_root_path": "/home/tvlad/Projects/MasterThesis/LiTS/experiments_data/set_4_1/",
        "carthesian_model_results_dirname": "experiment_1",
        "polar_model_results_dirname": "experiment_2"
        },
    "DeepLabV3+": {
        "results_root_path": "/home/tvlad/Projects/MasterThesis/LiTS/experiments_data/set_6_1/",
        "carthesian_model_results_dirname": "experiment_1",
        "polar_model_results_dirname": "experiment_2"
        }
    }

with st.expander("Filters"):
    model_arch = st.selectbox("Model Architecture", list(available_models.keys()))
    test_dataset = st.selectbox("Evaluation Dataset", available_datasets)
    test_volume = st.selectbox("Volume", available_datasets[test_dataset])

TEST_RESULTS_DIRNAME = f"test_results_{test_dataset}_Positive_viz"
model_dict = available_models[model_arch]

carthesian_results_root_dirpath = os.path.join(model_dict["results_root_path"], model_dict["carthesian_model_results_dirname"], TEST_RESULTS_DIRNAME, test_volume)
polar_results_root_path = os.path.join(model_dict["results_root_path"], model_dict["polar_model_results_dirname"], TEST_RESULTS_DIRNAME, test_volume)
carthesian_results_entries = os.listdir(carthesian_results_root_dirpath)
polar_results_entries = os.listdir(polar_results_root_path)

if len(carthesian_results_entries) != len(polar_results_entries):
    raise Exception("Inconsistent number of results")

first_carthesian_slice_idx = int(sorted(os.listdir(carthesian_results_root_dirpath), key=lambda dir_entry: int(dir_entry.split("-")[-1].split(".")[0]))[0].split("-")[-1].split(".")[0])
last_carthesian_slice_idx = int(sorted(os.listdir(carthesian_results_root_dirpath), key=lambda dir_entry: int(dir_entry.split("-")[-1].split(".")[0]))[-1].split("-")[-1].split(".")[0])

first_polar_slice_idx = int(sorted(os.listdir(polar_results_root_path), key=lambda dir_entry: int(dir_entry.split("-")[-1].split(".")[0]))[0].split("-")[-1].split(".")[0])
last_polar_slice_idx = int(sorted(os.listdir(polar_results_root_path), key=lambda dir_entry: int(dir_entry.split("-")[-1].split(".")[0]))[-1].split("-")[-1].split(".")[0])

if first_carthesian_slice_idx != first_polar_slice_idx or last_carthesian_slice_idx != last_polar_slice_idx:
    raise Exception("Slice indices don't match. Retry.")

first_slice_idx, last_slice_idx = st.select_slider(
    "Select Volume Scan Slice Range",
    options=list(range(first_carthesian_slice_idx, last_carthesian_slice_idx)),
    value=(first_carthesian_slice_idx, first_carthesian_slice_idx + 1)
)

for slice_idx in range(first_slice_idx, last_slice_idx):
    slice_name = f"slice-{slice_idx}.png"
    
    if slice_name in carthesian_results_entries and slice_name in polar_results_entries:
        carthesian_res = cv.imread(os.path.join(carthesian_results_root_dirpath, slice_name))
        polar_res = cv.imread(os.path.join(polar_results_root_path, slice_name))

        st.image(
            image=[carthesian_res, polar_res],
            caption=["Carthesian Model", "Polar Model"],
            channels="BGR",
            width=860
            )
