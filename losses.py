import torch
import torch.nn as nn
import torch.nn.functional as F
import config
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
l2loss = nn.MSELoss()


def l2normalize(x):
    return F.normalize(x, p=2, dim=1)


class CosAttentionLoss(nn.Module):
    """
    CosAttention loss
    Regress the cosine using weight-averaged patched cosine
    """

    def __init__(self):
        super(CosAttentionLoss, self).__init__()

    def computeXCos(self, feat_grid_1, feat_grid_2, attention):
        attention = attention.permute(0, 2, 3, 1)
        feat1 = feat_grid_1.permute(0, 2, 3, 1)
        feat2 = feat_grid_2.permute(0, 2, 3, 1)
        feat1 = feat1.contiguous().view(-1, feat1.size(3))
        feat2 = feat2.contiguous().view(-1, feat2.size(3))
        feat1 = l2normalize(feat1)
        feat2 = l2normalize(feat2)
        cos_patched = cos(feat1, feat2).view_as(attention)
        cos_attentioned = (cos_patched * attention)
        # cos_a:  torch.Size([bs, 7, 7, 1]) ->[bs, 49]
        cos_attentioned = cos_attentioned.view(cos_attentioned.size(0), -1)
        cos_attentioned = cos_attentioned.sum(1)
        return cos_attentioned

    def forward(self, feat_grid_1, feat_grid_2, attention,
                cos_tgt, size_average=True):
        '''
        feat_grid_1.size(): [bs, 32, 7, 7]
        feat1/ feat2: [bs, c, 7, 7] -> [bs, 7, 7, c] -> [bs * 7*7, c]
        attention:    [bs, 1, 7, 7] -> [bs, 7, 7, 1]
        cos_tgt: [bs]
        '''
        cos_attentioned = self.computeXCos(feat_grid_1, feat_grid_2, attention)
        losses = l2loss(cos_attentioned, cos_tgt)

        losses = losses.mean() if size_average else losses.sum()

        return losses


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)
