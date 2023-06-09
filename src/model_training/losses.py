import torch
from model_training.constants import EPSILON

class MultiClassPixelWiseCrossEntropy(torch.nn.Module):
    """
    Pixel-Weighted Multiclass CrossEntropyLoss.
    """
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.wmap_weight = self.config["training_config"]["wmap_weight"]
        weight = torch.Tensor(self.config["training_config"]["class_weights"]).to(self.config["device"])
        self.loss = torch.nn.CrossEntropyLoss(weight=weight, reduction='none')
        self.require_weightmaps = True if self.wmap_weight else False
        self.require_one_hot = False
        self.require_single_channel_mask = True
        self.require_single_channel_input = False

    def forward(self, inp, target, weight_map=None):
        batch_size, sample_channels = inp.size()[:2]

        inp = inp.view(batch_size, sample_channels,-1).type(torch.FloatTensor).to(self.config["device"])
        target = target.view(batch_size,-1).type(torch.LongTensor).to(self.config["device"])
        weight_map = weight_map.view(batch_size,-1).type(torch.FloatTensor).to(self.config["device"]) if weight_map is not None else torch.zeros_like(target).type(torch.FloatTensor).to(self.config["device"])

        loss = self.loss(inp, target)
        batch_loss = torch.mean(loss * (weight_map + 1.) ** self.wmap_weight)

        return batch_loss

class MultiClassDice(torch.nn.Module):
    """
    Surrogate loss for multiclass dice loss by computing the dice score per channel.
    """

    def __init__(self, config):
        super().__init__()

        self.weight_score = torch.tensor(config["training_config"]["weight_score"], dtype=torch.float, device=config["device"]) if config["training_config"]["weight_score"] is not None else None
        self.require_weightmaps = False
        self.require_one_hot = True
        self.require_single_channel_mask = False
        self.require_single_channel_input = False
        self.config = config

    def forward(self, inp, target_one_hot):
        batch_size, channels = inp.size()[:2]

        inp = inp.type(torch.FloatTensor).to(self.config["device"]).view(batch_size, channels, -1)
        target_one_hot = target_one_hot.type(torch.FloatTensor).to(self.config["device"]).view(batch_size, channels, -1)

        intersection = torch.sum(inp * target_one_hot, dim=2)
        union = torch.sum(inp, dim=2) + torch.sum(target_one_hot, dim=2) + EPSILON

        if self.weight_score is not None:
            weight_score = torch.stack([self.weight_score for _ in range(batch_size)], dim=0)
            dice_loss = torch.mean(-1. * torch.mean(2. * intersection * weight_score / union, dim=1))
        else:
            dice_loss = torch.mean(-1. * torch.mean(2. * intersection / union, dim=1))

        return dice_loss

class MultiClassCombined(torch.nn.Module):
    """
    Pixel-Weighted Multiclass CrossEntropyLoss over Multiclass Dice Loss
    """

    def __init__(self, config):
        super().__init__()

        self.wmap_weight = config["training_config"]["wmap_weight"]
        self.cross_entropy_loss = MultiClassPixelWiseCrossEntropy(config=config)
        self.dice_loss = MultiClassDice(config=config)
        self.require_weightmaps = True if self.wmap_weight else None
        self.require_one_hot = self.dice_loss.require_one_hot or self.cross_entropy_loss.require_one_hot
        self.require_single_channel_mask = self.dice_loss.require_single_channel_mask or self.cross_entropy_loss.require_single_channel_mask
        self.require_single_channel_input = self.dice_loss.require_single_channel_input or self.cross_entropy_loss.require_single_channel_input

    def forward(self, inp, target, target_one_hot, weight_map=None):
        pixel_wise_cross_entropy_loss = self.cross_entropy_loss(inp, target, weight_map)
        dice_loss = self.dice_loss(inp, target_one_hot)
        combined_loss = pixel_wise_cross_entropy_loss / (-1 * dice_loss)

        return combined_loss
