import numpy as np
from model_training.metrics import Metrics
from dataset_preparation.ircadb_dataset import IRCADB_Dataset
from dataset_preparation.acadtum_dataset import ACADTUM_Dataset
import onnx, onnxruntime, torch, os, argparse, csv, tqdm
from model_training.preprocessing_utils import centroid, to_polar, to_cart

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_filename", type=str, default="test_metrics.csv")
    parser.add_argument("--carthesian_experiment_path", type=str, default="./LiTS/experiments_data/set_6_1/experiment_1/")
    parser.add_argument("--polar_experiment_path", type=str, default="./LiTS/experiments_data/set_6_1/experiment_2/")
    parser.add_argument("--mode", type=str, default="polar")   # carthesian or polar
    parser.add_argument("--output_directory_path", type=str, default="./LiTS/experiments_data/set_6_1/experiment_2/test_results_ACADTUM_Positive/")
    parser.add_argument("--dataset", type=str, default="ACADTUM")  # 3DIRCADB, ACADTUM

    return parser.parse_args()

def load_experiment_inference_session(experiment_root_path):
    MODEL_NAME = "best_val_dice.onnx"

    carthesian_model = onnx.load(os.path.join(experiment_root_path, MODEL_NAME))
    onnx.checker.check_model(carthesian_model)

    inference_session = onnxruntime.InferenceSession(os.path.join(experiment_root_path, MODEL_NAME))

    return inference_session

def run_inference_on_sample(inference_session, input_slice):
    model_probabilities = inference_session.run(None, {"input": input_slice})[0]
    model_prediction = np.argmax(model_probabilities, axis=1)
    return model_prediction, model_probabilities

def polar_inference(carthesian_inf_session, polar_inf_session, scan_slice):
    # Compute carthesian model prediction
    carthesian_model_prediction, _ = run_inference_on_sample(carthesian_inf_session, scan_slice)
    
    # Extract center from carthesian prediction
    center = centroid(carthesian_model_prediction[0].astype(np.uint8))

    # Convert input scan slice to polar coordinate system
    polar_scan_slice = np.expand_dims(np.expand_dims(to_polar(scan_slice[0,0], center), 0), 0).astype(np.float32)

    # Compute polar model prediction
    polar_model_prediction, polar_model_probabilities = run_inference_on_sample(polar_inf_session, polar_scan_slice)
    polar_model_prediction = np.expand_dims(to_cart(polar_model_prediction[0], center), axis=0)
    polar_model_probabilities = np.expand_dims(np.stack([to_cart(polar_model_probabilities[0, 0], center), to_cart(polar_model_probabilities[0, 1], center)], axis=0), 0)

    return polar_model_prediction, polar_model_probabilities

if __name__ == "__main__":
    # Load test dataset and create dataloader

    script_arguments = parse_arguments()

    if script_arguments.dataset == "3DIRCADB":
        test_dataset = IRCADB_Dataset(root_path="../../Downloads/test_dataset_3DIRCADB/Test_Data_3Dircadb1/")
    elif script_arguments.dataset == "ACADTUM":
        test_dataset = ACADTUM_Dataset(root_path="../../Downloads/test_dataset_ACADTUM")
    else:
        raise NotImplementedError(f"Unknown argument: {script_arguments.dataset}")
    
    test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False
        )

    # Load carthesian and polar inference sessions
    if script_arguments.mode == "carthesian":
        carthesian_inference_session = load_experiment_inference_session(script_arguments.carthesian_experiment_path)
    elif script_arguments.mode == "polar":
        carthesian_inference_session = load_experiment_inference_session(script_arguments.carthesian_experiment_path)
        polar_inference_session = load_experiment_inference_session(script_arguments.polar_experiment_path)
    else:
        raise NotImplementedError

    # Prepare auxiliaries
    iter_preds_collect, iter_target_collect, iter_probs_collect = [], [], []

    if not os.path.exists(script_arguments.output_directory_path):
        os.mkdir(script_arguments.output_directory_path)

    # Prepare log file
    results_log = open(os.path.join(script_arguments.output_directory_path, script_arguments.results_filename), "w", encoding="UTF-8")
    test_metrics_ledger = csv.writer(results_log)
    test_metrics_ledger.writerow(["volume", "test_dice", "test_iou", "test_precision", "test_accuracy", "test_recall", "test_specificity", "test_auc_score"])



    for item_idx , test_item in tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        input_image = test_item["input_image"].numpy()
        input_mask = test_item["input_mask"].numpy()
        
        masks_save_dir_path = os.path.join(script_arguments.output_directory_path, test_item["volume"][0])
        if not os.path.exists(masks_save_dir_path):
            os.mkdir(masks_save_dir_path)

        if script_arguments.mode == "carthesian":
            predicted_mask, predicted_probabilities = run_inference_on_sample(carthesian_inference_session, input_image)
        elif script_arguments.mode == "polar":
            predicted_mask, predicted_probabilities = polar_inference(carthesian_inference_session, polar_inference_session, input_image)
        else:
            raise NotImplementedError
        
        np.save(os.path.join(masks_save_dir_path, test_item["slice_path"][0].split("/")[-1]), predicted_mask[0])
        
        iter_target_collect.append(input_mask.astype(np.uint8))
        iter_preds_collect.append(predicted_mask.astype(np.uint8))
        iter_probs_collect.append(predicted_probabilities)

        if test_item["volume_change"] == True or item_idx == len(test_dataloader) - 1:  
            # print(f"Compute metrics for {test_item['volume'][0]} at {test_item['slice_path'][0].split('/')[-1]}")
            
            probabilities_predictions = np.vstack(iter_probs_collect)
            class_predictions = np.vstack(iter_preds_collect)
            labels = np.vstack(iter_target_collect)

            iter_preds_collect, iter_target_collect, iter_probs_collect = [], [], []

            true_negatives, false_positives, false_negatives, true_positives = Metrics.ConfusionMatrix(class_predictions, labels).ravel()
            
            mini_dice = Metrics.Dice2(true_positives, false_positives, false_negatives)
            mini_iou = Metrics.IoU2(true_positives, false_positives, false_negatives)
            mini_accuracy = Metrics.Accuracy(true_positives, true_negatives, false_positives, false_negatives)
            mini_precision = Metrics.Precision(true_positives, false_positives)
            mini_recall = Metrics.Recall(true_positives, false_negatives)
            mini_specificity = Metrics.Specificity(true_negatives, false_positives)
            mini_auc_score = Metrics.AuC(labels, probabilities_predictions)

            test_metrics_ledger.writerow([test_item['volume'][0], mini_dice, mini_iou, mini_precision, mini_accuracy, mini_recall, mini_specificity, mini_auc_score])

    results_log.close()