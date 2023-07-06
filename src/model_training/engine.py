import torch
import numpy as np
from tqdm import tqdm
from model_training.metrics import Metrics
from torcheval.metrics.functional import binary_confusion_matrix, binary_auroc

def model_trainer(model_setup, data_loader, loss_func, device, metrics_idx, metrics, epoch):
    model, optimizer = model_setup
    _ = model.train()

    iter_preds_collect, iter_target_collect, iter_loss_collect, iter_probs_collect = [], [], [], []
    epoch_loss_collect, epoch_dice_collect, epoch_iou_collect,\
    epoch_precision_collect, epoch_recall_collect, epoch_specificity_collect,\
    epoch_accuracy_collect, epoch_auc_collect = [], [], [], [], [], [], [], []

    train_data_iter = tqdm(data_loader, position=2)
    inp_string = 'Epoch {}'.format(epoch)

    for slice_idx, file_dict in enumerate(train_data_iter):
        if np.count_nonzero(file_dict["targets"]) == 0:
            raise Exception("Found unexpected negative sample to train with")

        train_data_iter.set_description(inp_string)

        training_slice  = file_dict["input_images"].type(torch.FloatTensor).to(device)
        training_mask = file_dict["targets"].type(torch.FloatTensor).to(device)

        ### GET PREDICTION ###
        model_output = model(training_slice)

        ### BASE LOSS ###
        feed_dict = {'inp':model_output}
        if loss_func.require_single_channel_mask:
            feed_dict['target'] = file_dict['targets'].to(device)
        if loss_func.require_one_hot:
            feed_dict['target_one_hot'] = file_dict['one_hot_targets'].to(device)
        if loss_func.require_weightmaps:
            feed_dict['weight_map'] = file_dict["weightmaps"].to(device)
        
        loss = loss_func(**feed_dict)

        ### RUN BACKPROP AND WEIGHT UPDATE ###
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### GET SCORES ###
        max_predicted_probs, max_predicted_probs_idx = torch.max(model_output, dim=1)
        iter_probs_collect.append(max_predicted_probs)
        iter_preds_collect.append(max_predicted_probs_idx)
        iter_target_collect.append(training_mask)
        iter_loss_collect.append(loss.item())
        
        ### Compute Metrics of collected training samples
        if slice_idx % metrics_idx == 0 and slice_idx != 0:
            probabilities_predictions = torch.vstack(iter_probs_collect).flatten()
            class_predictions = torch.vstack(iter_preds_collect).flatten()
            labels = torch.vstack(iter_target_collect).flatten()

            true_negatives, false_positives, false_negatives, true_positives = binary_confusion_matrix(class_predictions, labels.type(torch.int)).cpu().numpy().ravel()
            
            dice_score = Metrics.Dice2(true_positives, false_positives, false_negatives)
            iou_score = Metrics.IoU2(true_positives, false_positives, false_negatives)

            accuracy = Metrics.Accuracy(true_positives, true_negatives, false_positives, false_negatives)
            precision = Metrics.Precision(true_positives, false_positives)
            recall = Metrics.Recall(true_positives, false_negatives)
            specificity = Metrics.Specificity(true_negatives, false_positives)

            mini_auc_score = binary_auroc(probabilities_predictions, labels).cpu().numpy()

            epoch_accuracy_collect.append(accuracy)
            epoch_dice_collect.append(dice_score)
            epoch_iou_collect.append(iou_score)
            epoch_precision_collect.append(precision)
            epoch_recall_collect.append(recall)
            epoch_specificity_collect.append(specificity)
            epoch_auc_collect.append(mini_auc_score)
            epoch_loss_collect.append(np.mean(iter_loss_collect))
            
            ### Add Scores to metric collector
            ### Update tqdm string
            inp_string = 'Epoch {0} || Loss: {1:3.7f} | Dice: {2:2.5f}'.format(epoch, np.mean(epoch_loss_collect), np.mean(epoch_dice_collect))

            ### Reset mini collector lists
            iter_preds_collect, iter_target_collect, iter_loss_collect, iter_probs_collect = [], [], [], []

    metrics["train_dice"].append(np.mean(epoch_dice_collect))
    metrics["train_iou"].append(np.mean(epoch_iou_collect))
    metrics["train_accuracy"].append(np.mean(epoch_accuracy_collect))
    metrics["train_precision"].append(np.mean(epoch_precision_collect))
    metrics["train_recall"].append(np.mean(epoch_recall_collect))
    metrics["train_specificity"].append(np.mean(epoch_recall_collect))
    metrics["train_auc_score"].append(np.mean(epoch_auc_collect))
    metrics["train_loss"].append(np.mean(epoch_loss_collect))

@torch.no_grad()
def model_validator(model, data_loader, loss_func, device, num_classes, metrics, epoch):
    _ = model.eval()

    iter_preds_collect, iter_target_collect, iter_probs_collect, iter_loss_collect = [], [], [], []
    epoch_loss_collect, epoch_dice_collect, epoch_iou_collect,\
    epoch_precision_collect, epoch_recall_collect, epoch_specificity_collect,\
    epoch_accuracy_collect, epoch_auc_collect = [], [], [], [], [], [], [], []

    validation_data_iter = tqdm(data_loader, position=2)
    inp_string = 'Epoch {}'.format(epoch)

    for slice_idx, file_dict in enumerate(validation_data_iter):
        if np.count_nonzero(file_dict["targets"]) == 0:
            raise Exception("Found unexpected negative sample to validate with")

        validation_data_iter.set_description(inp_string)

        ### GET PREDICTION ###
        validation_slice = file_dict["input_images"].type(torch.FloatTensor).to(device)
        validation_mask = file_dict["targets"].type(torch.FloatTensor).to(device)
        model_output = model(validation_slice)

        ### GET SCORES ###
        max_predicted_probs, max_predicted_probs_idx = torch.max(model_output, dim=1)
        iter_probs_collect.append(max_predicted_probs)
        iter_preds_collect.append(max_predicted_probs_idx)
        iter_target_collect.append(validation_mask)

        feed_dict = {'inp':model_output}
        if loss_func.require_single_channel_mask:
            feed_dict['target'] = file_dict['targets'].to(device)
        if loss_func.require_one_hot:
            feed_dict['target_one_hot'] = file_dict['one_hot_targets'].to(device)
        if loss_func.require_weightmaps:
            feed_dict['weight_map'] = file_dict["weightmaps"].to(device)
        loss = loss_func(**feed_dict)
        iter_loss_collect.append(loss.item())

        if file_dict['vol_change'] or slice_idx == len(data_loader) - 1:

            probabilities_predictions = torch.vstack(iter_probs_collect).flatten()
            class_predictions = torch.vstack(iter_preds_collect).flatten()
            labels = torch.vstack(iter_target_collect).flatten()

            true_negatives, false_positives, false_negatives, true_positives = binary_confusion_matrix(class_predictions, labels.type(torch.int)).cpu().numpy().ravel()

            mini_dice = Metrics.Dice2(true_positives, false_positives, false_negatives)
            mini_iou = Metrics.IoU2(true_positives, false_positives, false_negatives)
            
            mini_accuracy = Metrics.Accuracy(true_positives, true_negatives, false_positives, false_negatives)
            mini_precision = Metrics.Precision(true_positives, false_positives)
            mini_recall = Metrics.Recall(true_positives, false_negatives)
            mini_specificity = Metrics.Specificity(true_negatives, false_positives)
            
            mini_auc_score = binary_auroc(probabilities_predictions, labels).cpu().numpy()

            epoch_loss_collect.append(np.mean(iter_loss_collect))
            epoch_dice_collect.append(mini_dice)
            epoch_iou_collect.append(mini_iou)
            epoch_accuracy_collect.append(mini_accuracy)
            epoch_precision_collect.append(mini_precision)
            epoch_recall_collect.append(mini_recall)
            epoch_auc_collect.append(mini_auc_score)
            epoch_specificity_collect.append(mini_specificity)

            inp_string = 'Epoch {0} || Loss: {1:2.5f}/Vol | Dice: {2:2.5f}/Vol'.format(epoch, np.mean(epoch_loss_collect), np.mean(epoch_dice_collect))
            validation_data_iter.set_description(inp_string)
            iter_loss_collect, iter_preds_collect, iter_target_collect, iter_probs_collect = [], [], [], []

    metrics['val_dice'].append(np.mean(epoch_dice_collect))
    metrics['val_iou'].append(np.mean(epoch_iou_collect))
    metrics["val_accuracy"].append(np.mean(epoch_accuracy_collect))
    metrics['val_precision'].append(np.mean(epoch_precision_collect))
    metrics['val_recall'].append(np.mean(epoch_recall_collect))
    metrics['val_specificity'].append(np.mean(epoch_specificity_collect))
    metrics["val_auc_score"].append(np.mean(epoch_auc_collect))
    metrics['val_loss'].append(np.mean(epoch_loss_collect))
