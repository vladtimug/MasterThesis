import torch
from metrics import PixelAccuracy

class Engine():
    def train_batch(model, data, optimizer, criterion):
        model.train()

        images, annotations = data
        predictions = model(images)
        optimizer.zero_grad()

        loss = criterion(predictions, annotations)
        accuracy = PixelAccuracy(predictions, annotations)

        loss.backward()
        optimizer.step()

        return loss.item(), accuracy.item()

    @torch.no_grad()
    def validate_bacth(model, data, criterion):
        model.eval()

        images, annotations = data
        predictions = model(images)

        loss = criterion(predictions, annotations)
        accuracy = PixelAccuracy(predictions, annotations)

        return loss.item(), accuracy.item()