import torch

cross_entropy_loss = torch.nn.CrossEntropyLoss()

def CrossEntropy(predictions, targets):
    batch_loss = cross_entropy_loss(predictions, targets)
    return batch_loss