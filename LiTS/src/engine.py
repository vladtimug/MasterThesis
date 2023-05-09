import time
import torch
import numpy as np
from tqdm import tqdm
from metrics import Dice, IoU, ConfusionMatrix, Precision, Recall, Specificity, Accuracy, AuC

def model_trainer(model_setup, data_loader, loss_func, device, metrics_idx, metrics, epoch):
    model, optimizer = model_setup
    _ = model.train()

    # base_loss_func, aux_loss_func = losses

    iter_preds_collect, iter_target_collect, iter_loss_collect, iter_probs_collect = [], [], [], []
    epoch_loss_collect, epoch_dice_collect, epoch_iou_collect,\
    epoch_precision_collect, epoch_recall_collect, epoch_specificity_collect,\
    epoch_accuracy_collect, epoch_auc_collect = [], [], [], [], [], [], [], []

    train_data_iter = tqdm(data_loader, position=2)
    inp_string = 'Epoch {}'.format(epoch)

    for slice_idx, file_dict in enumerate(train_data_iter):
        train_data_iter.set_description(inp_string)

        train_iter_start_time = time.time()

        training_slice  = file_dict["input_images"].type(torch.FloatTensor).to(device)

        ### GET PREDICTION ###
        model_output, _ = model(training_slice)

        ### BASE LOSS ###
        feed_dict = {'inp':model_output}
        if loss_func.require_single_channel_mask:
            feed_dict['target'] = file_dict['targets'].to(device)
        if loss_func.require_one_hot:
            feed_dict['target_one_hot'] = file_dict['one_hot_targets'].to(device)
        if loss_func.require_weightmaps:
            feed_dict['weight_map'] = file_dict["weightmaps"].to(device)
        
        loss_base = loss_func(**feed_dict)

        ### AUXILIARY LOSS ###
        # loss_aux = torch.tensor(0).type(torch.FloatTensor).to(device)
        # if opt.Network['use_auxiliary_inputs']:
        #     for aux_ix in range(len(auxiliaries)):
        #         feed_dict = {'inp':auxiliaries[aux_ix]}
        #         if aux_loss_func.loss_func.require_weightmaps:
        #             feed_dict['wmap'] = file_dict['aux_weightmaps'][aux_ix].to(device)
        #         if aux_loss_func.loss_func.require_one_hot:
        #             feed_dict['target_one_hot'] = file_dict['one_hot_aux_targets'][aux_ix].to(device)
        #         if aux_loss_func.loss_func.require_single_channel_mask:
        #             feed_dict['target'] = file_dict['aux_targets'][aux_ix].to(device)

        #         loss_aux = loss_aux + 1./(aux_ix+1)*aux_loss_func(**feed_dict)

        ### COMBINE LOSS FUNCTIONS ###
        # loss = loss_base+loss_aux
        loss = loss_base

        ### RUN BACKPROP AND WEIGHT UPDATE ###
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_probs_collect.append(model_output.detach().cpu().numpy())

        ### GET SCORES ###
        if model_output.shape[1]!=1:
            iter_preds_collect.append(np.argmax(model_output.detach().cpu().numpy(),axis=1))
        else:
            iter_preds_collect.append(np.round(model_output.detach().cpu().numpy()))

        iter_target_collect.append(file_dict['targets'].numpy())
        iter_loss_collect.append(loss.item())

        if slice_idx % metrics_idx == 0 and slice_idx != 0:
            ### Compute Dice Score of collected training samples
            dice_score = Dice(np.vstack(iter_preds_collect), np.vstack(iter_target_collect))
            iou_score = IoU(np.vstack(iter_preds_collect), np.vstack(iter_target_collect))

            confusion_matrix = ConfusionMatrix(np.vstack(iter_preds_collect), np.vstack(iter_target_collect))
            false_positives, true_positives = confusion_matrix[0 ,1], confusion_matrix[1, 1]
            false_negatives, true_negatives = confusion_matrix[1 ,0], confusion_matrix[0, 0]
            
            accuracy = Accuracy(true_positives, true_negatives, false_positives, false_negatives)
            precision = Precision(true_positives, false_positives)
            recall = Recall(true_positives, false_negatives)
            specificity = Specificity(true_negatives, false_positives)

            mini_auc_score = AuC(np.vstack(iter_target_collect), np.vstack(iter_probs_collect))

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

    iter_preds_collect, iter_target_collect, iter_probs_collect = [], [], []
    epoch_loss_collect, epoch_dice_collect, epoch_iou_collect,\
    epoch_precision_collect, epoch_recall_collect, epoch_specificity_collect,\
    epoch_accuracy_collect, epoch_auc_collect = [], [], [], [], [], [], [], []

    validation_data_iter = tqdm(data_loader, position=2)
    inp_string = 'Epoch {}'.format(epoch)

    for slice_idx, file_dict in enumerate(validation_data_iter):
        validation_data_iter.set_description(inp_string)

        validation_slice = file_dict["input_images"].type(torch.FloatTensor).to(device)
        validation_mask  = file_dict["targets"].to(device)
        
        if 'crop_option' in file_dict.keys():
            validation_crop = file_dict['crop_option'].type(torch.FloatTensor).to(device)
        
        model_output = model(validation_slice)[0]

        if 'crop_option' in file_dict.keys():
            model_output = model_output * validation_crop

        iter_probs_collect.append(model_output.detach().cpu().numpy())

        if num_classes != 1:
            model_prediction = np.argmax(model_output.detach().cpu().numpy(), axis=1)
        else:
            model_prediction = np.round(model_output.detach().cpu().numpy())

        iter_preds_collect.append(model_prediction)
        
        iter_target_collect.append(validation_mask.detach().cpu().numpy())

        if file_dict['vol_change'] or slice_idx == len(data_loader) - 1:
            feed_dict = {'inp':model_output}
            if loss_func.require_single_channel_mask:
                feed_dict['target'] = file_dict['targets'].to(device)
            if loss_func.require_one_hot:
                feed_dict['target_one_hot'] = file_dict['one_hot_targets'].to(device)
            if loss_func.require_weightmaps:
                feed_dict['weight_map'] = file_dict["weightmaps"].to(device)
            loss = loss_func(**feed_dict)
            epoch_loss_collect.append(loss.item())

            mini_dice = Dice(np.vstack(iter_preds_collect), np.vstack(iter_target_collect))
            mini_iou = IoU(np.vstack(iter_preds_collect), np.vstack(iter_target_collect))
            
            confusion_matrix = ConfusionMatrix(np.vstack(iter_preds_collect), np.vstack(iter_target_collect))
            false_positives, true_positives = confusion_matrix[0 ,1], confusion_matrix[1, 1]
            false_negatives, true_negatives = confusion_matrix[1 ,0], confusion_matrix[0, 0]
            
            mini_accuracy = Accuracy(true_positives, true_negatives, false_positives, false_negatives)
            mini_precision = Precision(true_positives, false_positives)
            mini_recall = Recall(true_positives, false_negatives)
            mini_specificity = Specificity(true_negatives, false_positives)
            
            mini_auc_score = AuC(np.vstack(iter_preds_collect), np.vstack(iter_probs_collect))

            epoch_dice_collect.append(mini_dice)
            epoch_iou_collect.append(mini_iou)
            epoch_accuracy_collect.append(mini_accuracy)
            epoch_precision_collect.append(mini_precision)
            epoch_recall_collect.append(mini_recall)
            epoch_auc_collect.append(mini_auc_score)
            epoch_specificity_collect.append(mini_specificity)

            inp_string = 'Epoch {0} || Loss: {1:2.5f}/Vol | Dice: {1:2.5f}/Vol'.format(epoch, np.mean(epoch_loss_collect), np.mean(epoch_dice_collect))
            validation_data_iter.set_description(inp_string)
            iter_preds_collect, iter_target_collect, iter_probs_collect = [], [], []

    metrics['val_dice'].append(np.mean(epoch_dice_collect))
    metrics['val_iou'].append(np.mean(epoch_iou_collect))
    metrics["val_accuracy"].append(np.mean(epoch_accuracy_collect))
    metrics['val_precision'].append(np.mean(epoch_precision_collect))
    metrics['val_recall'].append(np.mean(epoch_recall_collect))
    metrics['val_specificity'].append(np.mean(epoch_specificity_collect))
    metrics["val_auc_score"].append(np.mean(epoch_auc_collect))
    metrics['val_loss'].append(np.mean(epoch_loss_collect))
