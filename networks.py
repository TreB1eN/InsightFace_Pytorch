import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionXCosNet(nn.Module):
    def __init__(self, conf):
        super(AttentionXCosNet, self).__init__()
        self.embedding_net = nn.Sequential(
                    nn.Conv2d(32, 16, 3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.PReLU()
                )
        self.attention = nn.Sequential(
                    nn.Conv2d(32, 16, 3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.PReLU(),
                    nn.Conv2d(16, 1, 3, padding=1),
                    nn.BatchNorm2d(1),
                    nn.PReLU(),
                )
        self.name = 'AttenCosNet'
        self.USE_SOFTMAX = conf.USE_SOFTMAX
        self.SOFTMAX_T = conf.SOFTMAX_T

    def softmax(self, x, T=1):
        x /= T
        return F.softmax(x.reshape(x.size(0), x.size(1), -1), 2).view_as(x)

    def divByNorm(self, x):
        '''
            attention_weights.size(): [bs, 1, 7, 6]
        '''
        x -= x.view(x.size(0), x.size(1), -1).min(dim=2)[0].repeat(1, 1, x.size(2) * x.size(3)).view(x.size(0), x.size(1), x.size(2), x.size(3))
        x /=  x.view(x.size(0), x.size(1), -1).sum(dim=2).repeat(1, 1, x.size(2) * x.size(3)).view(x.size(0), x.size(1), x.size(2), x.size(3))
        return x

    def forward(self, feat_grid_1, feat_grid_2):
        '''
            feat_grid_1.size(): [bs, 32, 7, 7]
            attention_weights.size(): [bs, 1, 7, 7]
        '''
        # XXX Do I need to normalize grid_feat?
        conv1 = self.embedding_net(feat_grid_1)
        conv2 = self.embedding_net(feat_grid_2)
        fused_feat = torch.cat((conv1, conv2), dim=1)
        attention_weights = self.attention(fused_feat)
        # To Normalize attention
        if self.USE_SOFTMAX:
            attention_weights = self.softmax(attention_weights, self.SOFTMAX_T)
        else:
            attention_weights = self.divByNorm(attention_weights)
        return attention_weights


class AttentionCosNet(nn.Module):
    def __init__(self):
        super(AttentionCosNet, self).__init__()
        self.embedding_net = nn.Sequential(
                    nn.Conv2d(512, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.PReLU()
                )
        self.attention = nn.Sequential(
                    nn.Conv2d(512, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.PReLU(),
                    nn.Conv2d(256, 1, 3, padding=1),
                    nn.BatchNorm2d(1),
                    nn.PReLU(),
                )
        self.name = 'AttentionCosNet'
    def softmax(self, x):
        return F.softmax(x.reshape(x.size(0), x.size(1), -1), 2).view_as(x)

    def forward(self, x1, x2):
        '''
            x1.size(): [bs, 512, 7, 6]
            attention_weights.size(): [bs, 1, 7, 6]
        '''

        conv1 = self.embedding_net(x1)
        conv2 = self.embedding_net(x2)
        fused_feat = torch.cat((conv1, conv2), dim=1)
        attention_weights = self.attention(fused_feat)
        # XXX: I use softmax instead of normalize
        # attention_weights = F.normalize(attention_weights, p=2, dim=1)
        attention_weights = self.softmax(attention_weights)
        return x1, x2, attention_weights


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


class ENMSiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(ENMSiameseNet, self).__init__()
        self.embedding_net = embedding_net
        self.name = 'Siamese'

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class ENMTripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(ENMTripletNet, self).__init__()
        self.embedding_net = embedding_net
        self.name = 'Triplet'
    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


class ENMEmbeddingNet(nn.Module):
    def __init__(self):
        super(ENMEmbeddingNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(1024, 1024),
                                nn.PReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(1024, 1024),
                                nn.PReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(1024, 1024)
                                )
        self.name = 'ENMEmb'

    def forward(self, x):
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


