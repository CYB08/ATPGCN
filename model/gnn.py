import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from model.gnn_conv import GCNConv
from torch.nn import Dropout, Linear, BatchNorm1d

class PHAutoEncoder(nn.Module):
    def __init__(self, firland, midland, lasland):
        super(PHAutoEncoder, self).__init__()
        self.encode0 = nn.Linear(firland, midland)
        self.encode1 = nn.Linear(midland, lasland)
        self.decode0 = nn.Linear(lasland, midland)
        self.decode1 = nn.Linear(midland, firland)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    def forward(self, features):
        embedding = F.leaky_relu(self.encode0(features))
        mu = self.encode1(embedding)
        logvar = self.encode1(embedding)
        embedding_out = self.reparameterize(mu, logvar)
        embedding_res = F.leaky_relu(self.decode0(embedding_out))
        embedding_res = torch.sigmoid(self.decode1(embedding_res))
        return mu, logvar, embedding_out, embedding_res


class GcnMlp(nn.Module):
    def __init__(self, in_dim, mid_dim, las_dim, dropout):
        super(GcnMlp, self).__init__()
        self.fc1 = Linear(in_dim, mid_dim)
        self.fc2 = Linear(mid_dim, las_dim)
        self.Act1 = nn.ReLU()
        self.Act2 = nn.ReLU()
        self.reset_parameters()
        self.dropout = dropout
        self.BNorm0 = BatchNorm1d(in_dim, eps=1e-5)
        self.BNorm1 = BatchNorm1d(mid_dim, eps=1e-5)
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-5)
        nn.init.normal_(self.fc2.bias, std=1e-5)
    def forward(self, x):
        x = self.Act1(self.fc1(self.BNorm0(x)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.Act2(self.fc2(self.BNorm1(x)))
        return x


class MyGCN(torch.nn.Module):
    def __init__(self, infeat, outfeat, nclass, nROI, Land_dim=1000, Curve_dim=1000, nhid=32, dropout=0.8,
                 weight_decay=5e-4,
                 with_relu=True,
                 device=None):
        super(MyGCN, self).__init__()
        self.device = device
        self.nclass = nclass
        self.dropout = dropout
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        # MLP layer
        self.mid = 256
        self.last = 32
        # Graph Convolutational layer
        self.infeat = infeat
        self.nhid = 64
        self.outfeat = 16 #outfeat
        # ADE_Landscape layer
        self.firland  = Land_dim
        self.midland = 512
        self.lasland = 16
        # Conv_DervCurve layer
        self.fircurve = Curve_dim
        self.midcurve = 512
        self.lascurve = 32
        self.nROI = nROI

        self.VAEmodel = PHAutoEncoder(self.firland, self.midland, self.lasland)
        self.CurveTrans = nn.Sequential(nn.Linear(self.fircurve, self.midcurve, bias=True),
                                        nn.BatchNorm1d(self.midcurve),
                                        nn.LeakyReLU(), nn.BatchNorm1d(self.midcurve),
                                        nn.Linear(self.midcurve, self.lascurve, bias=True))
        self.conv1 = GCNConv(self.infeat, self.nhid, bias=True)
        self.conv2 = GCNConv(self.nhid, self.outfeat, bias=True)
        self.mlp = GcnMlp(self.nhid * 2 + self.outfeat * 2+ self.lasland + self.lascurve, self.mid, self.last, dropout)  # + self.lasland * 2 + self.lasland
        self.classifier = Linear(self.last, nclass)

    def forward(self, x, edge_index, batch, edge_attr, PH1_feat, PH0_feat):
        land_mu, land_logvar, land_embed, land_decoder = self.VAEmodel(PH1_feat)
        betti_curve = self.CurveTrans(PH0_feat)
        outfeat_PH = torch.cat([land_embed, betti_curve], dim=1)
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x_feat = torch.cat([x1, x2, outfeat_PH], dim=1)
        x_feat = self.mlp(x_feat)
        classfication = F.log_softmax(self.classifier(x_feat), dim=1)

        return classfication, land_mu, land_logvar, land_embed, land_decoder, betti_curve
