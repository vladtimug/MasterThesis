import torch
from torchmetrics.classification import MulticlassJaccardIndex, Dice

multiclass_iou = MulticlassJaccardIndex(
    num_classes=3,
    ignore_index=0,
    average="weighted"
).to("cuda")

multiclass_dice = Dice(
    num_classes=3,
    ignore_index=0,
    average="weighted"
).to("cuda")

def train_batch(model, data, optimizer, criterion):
    model.train()

    images, annotations = data
    predictions = model(images)
    optimizer.zero_grad()

    loss = criterion(predictions, annotations)
    
    diceScore = multiclass_dice(predictions, annotations)
    iouScore = multiclass_iou(predictions, annotations)

    loss.backward()
    optimizer.step()

    return loss.item(), diceScore.item(), iouScore.item()

@torch.no_grad()
def validate_batch(model, data, criterion):
    model.eval()

    images, annotations = data
    predictions = model(images)

    loss = criterion(predictions, annotations)
    
    diceScore = multiclass_dice(predictions, annotations)
    iouScore = multiclass_iou(predictions, annotations)

    return loss.item(), diceScore.item(), iouScore.item()