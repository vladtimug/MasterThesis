import torch

class MultiClassPixelWiseCrossEntropy(torch.nn.Module):
    """
    Pixel-Weighted Multiclass CrossEntropyLoss.
    """
    def __init__(self, config):
        super(MultiClassPixelWiseCrossEntropy, self).__init__()

        self.config = config

        self.wmap_weight   = self.config["training_config"]["wmap_weight"]

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

        # TODO: Figure out why 1 is added to the weight_map

        batch_loss = torch.mean(self.loss(inp, target) * (weight_map + 1.) ** self.wmap_weight)

        return batch_loss