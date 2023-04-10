import torch

def PixelAccuracy(predictions, targets):
    batch_accuracy = (torch.max(predictions, 1)[1] == targets).float().mean()
    return batch_accuracy