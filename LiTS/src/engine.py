import torch
from torchmetrics.functional import dice
from torchmetrics.classification import MulticlassJaccardIndex

multiclass_iou = MulticlassJaccardIndex(
    num_classes=3,
    ignore_index=0,
    average="weighted",
    mdmc_average="samplewise",
    threshold=0.9,
).to("cuda")

def train_batch(model, data, optimizer, criterion):
    model.train()

    images, annotations = data
    predictions = model(images)
    optimizer.zero_grad()

    loss = criterion(predictions, annotations)
    
    diceScore = dice(
            predictions, annotations,
            num_classes=3,
            ignore_index=0,
            average=None,
            mdmc_average="samplewise",
            threshold=0.9
        )
    backgroundDiceScore, liverDiceScore, tumorDiceScore = diceScore[0], diceScore[1], diceScore[2]

    iouScore = multiclass_iou(predictions, annotations)
    # backgroundIoUScore, liverIoUScore, tumorIoUScore = iouScore[0], iouScore[1], iouScore[2]

    loss.backward()
    optimizer.step()

    return loss.item(), \
            (backgroundDiceScore.item(), liverDiceScore.item(), tumorDiceScore.item()), \
            iouScore

@torch.no_grad()
def validate_batch(model, data, criterion):
    model.eval()

    images, annotations = data
    predictions = model(images)

    loss = criterion(predictions, annotations)

    diceScore = dice(
            predictions, annotations,
            num_classes=3,
            ignore_index=0,
            average="weighted",
            mdmc_average="samplewise",
            threshold=0.9
        )
    backgroundDiceScore, liverDiceScore, tumorDiceScore = diceScore[0], diceScore[1], diceScore[2]

    iouScore = multiclass_iou(predictions, annotations)
    # backgroundIoUScore, liverIoUScore, tumorIoUScore = iouScore[0], iouScore[1], iouScore[2]

    return loss.item(), \
            (backgroundDiceScore.item(), liverDiceScore.item(), tumorDiceScore.item()), \
            iouScore
