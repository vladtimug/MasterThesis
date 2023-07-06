import numpy as np
from sklearn.metrics import roc_curve, auc
from model_training.constants import EPSILON

class Metrics:
    @staticmethod
    def Dice(inp, target, eps=EPSILON):
        input_flatten = inp.flatten()
        target_flatten = target.flatten()
        overlap = np.sum(input_flatten * target_flatten)
        total_number_of_elements = np.sum(target_flatten)+np.sum(input_flatten)
        raw_dice_score = (2. * overlap) / (total_number_of_elements + eps)
        dice_score = np.clip(a=raw_dice_score, a_min=1e-4, a_max=0.9999)
        return dice_score

    @staticmethod
    def Dice2(true_positives, false_positives, false_negatives, eps=EPSILON):
        scaled_positives = 2 * true_positives
        raw_dice_score = scaled_positives / (scaled_positives + false_positives + false_negatives + eps)
        dice_score = np.clip(a=raw_dice_score, a_min=1e-4, a_max=0.9999)
        return dice_score

    @staticmethod
    def IoU(inp, target, eps=EPSILON):
        input_flatten = inp.flatten()
        target_flatten = target.flatten()
        intersection = np.sum(input_flatten * target_flatten)
        union = np.sum(input_flatten + target_flatten)
        raw_IoU = intersection / (union + eps)
        IoU = np.clip(a=raw_IoU, a_min=1e-4, a_max=0.9999)
        return IoU

    @staticmethod
    def IoU2(true_positives, false_positives, false_negatives, eps=EPSILON):
        raw_iou_score = true_positives / (true_positives + false_positives + false_negatives + eps)
        iou_score = np.clip(a=raw_iou_score, a_min=1e-4, a_max=0.9999)
        return iou_score

    @staticmethod
    def AuC(inp, target):
        max_prob = np.max(target, axis=1)

        input_flatten = inp.flatten()
        target_flatten = max_prob.flatten()

        fpr, tpr, _ = roc_curve(input_flatten, target_flatten)

        roc_auc_score = auc(fpr, tpr)

        return roc_auc_score

    @staticmethod
    def ConfusionMatrix(inp, target):
        input_flatten = inp.flatten()
        target_flatten = target.flatten()

        confusion_matrix = np.bincount(
            input_flatten.astype(int) * 2 + target_flatten.astype(int),
            minlength=4
        ).reshape((2, 2))
        return confusion_matrix

    @staticmethod
    def Accuracy(true_positives, true_negatives, false_positives, false_negatives):
        correct_predictions = true_positives + true_negatives
        total_predictions = true_positives + true_negatives + false_positives + false_negatives
        accuracy = np.round(correct_predictions / total_predictions, 3) 
        return accuracy

    @staticmethod
    def Specificity(true_negatives, false_positives, eps=EPSILON):
        total_negatives = true_negatives + false_positives
        specificity = np.round(true_negatives / (total_negatives + eps), 3)
        return specificity

    @staticmethod
    def Precision(true_positives, false_positives, eps=EPSILON):
        predicted_positives = true_positives + false_positives
        precision = np.round(true_positives / (predicted_positives + eps), 3)
        return precision

    @staticmethod
    def Recall(true_positives, false_negatives, eps=EPSILON):
        total_positives = true_positives + false_negatives
        recall = np.round(true_positives / (total_positives + eps), 3)
        return recall
