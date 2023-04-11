import torch
from torchmetrics.functional import dice
from torchmetrics.classification import MulticlassJaccardIndex

multiclass_iou = MulticlassJaccardIndex(
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
    
    diceScore = dice(
            predictions, annotations,
            average="weighted",
            mdmc_average="samplewise",
            ignore_index=0,
            num_classes=3
        )
    
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
    diceScore = dice(
            predictions, annotations,
            average="weighted",
            mdmc_average="samplewise",
            ignore_index=0,
            num_classes=3
        )
    
    iouScore = multiclass_iou(predictions, annotations)

    return loss.item(), diceScore.item(), iouScore.item()