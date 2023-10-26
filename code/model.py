import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn.conv import HypergraphConv
from torchmetrics.functional import pairwise_cosine_similarity as pss

def save_model(model, args):
    """Save model."""
    torch.save(model.state_dict(), args.outputPath + '/' + args.modelPath)
    print("Saved better model selected by validation.")
    return True

class HHGNN_hetero(nn.Module):
    """docstring for HHGNN_hetero"""
    def __init__(self, args):
        super(HHGNN_hetero, self).__init__()
        self.args = args

        # 3 different hyperConv, u_pp_a, u_pp, u_a
        # 2 layers
        self.hyper1 = nn.ModuleList([
            HypergraphConv(
                in_channels=args.hgcn_l1_in_channels,
                out_channels=args.hgcn_l1_out_channels,
                use_attention=args.hgcn_l1_use_attention,
                heads=args.hgcn_l1_heads,
                concat=args.hgcn_l1_concat,
                negative_slope=args.hgcn_l1_negative_slope,
                dropout=args.hgcn_l1_dropout,
                bias=args.hgcn_l1_bias,
                ) for _ in range(3)
            ])
        self.hyper2 = nn.ModuleList([
            HypergraphConv(
                in_channels=args.hgcn_l2_in_channels,
                out_channels=args.hgcn_l2_out_channels,
                use_attention=args.hgcn_l2_use_attention,
                heads=args.hgcn_l2_heads,
                concat=args.hgcn_l2_concat,
                negative_slope=args.hgcn_l2_negative_slope,
                dropout=args.hgcn_l2_dropout,
                bias=args.hgcn_l2_bias,
                ) for _ in range(3)
            ])

        # activation and dropout after each conv
        self.act1 = nn.Sequential(
            nn.LeakyReLU(args.hgcn_l1_after_leakySlope),
            nn.Dropout(p=args.model_dropout1),
            )
        self.act2 = nn.Sequential(
            nn.LeakyReLU(args.hgcn_l2_after_leakySlope),
            nn.Dropout(p=args.model_dropout2),
            )
        # linear and act before each conv
        self.layer0_linear = nn.ModuleList([nn.Linear(args.hgcn_l1_in_channels, args.hgcn_l1_in_channels) for _ in range(3)])
        self.layer0_act = nn.LeakyReLU(args.hgcn_l1_before_leakySlope)

        self.layer1_linear = nn.ModuleList([nn.Linear(args.hgcn_l1_out_channels, args.hgcn_l1_out_channels) for _ in range(3)])
        self.layer1_act = nn.LeakyReLU(args.hgcn_l2_before_leakySlope)

        # map g_dim to commonD
        self.g2c = nn.ModuleList([nn.Linear(args.hgcn_l2_out_channels, args.model_commonDim) for _ in range(3)])
        # map x_dim to commonD
        self.x2c = nn.ModuleList([nn.Linear(args.hgcn_l1_in_channels, args.model_commonDim) for _ in range(3)])

        # activate after commonD
        self.g2c_act = nn.LeakyReLU(args.model_leakySlope_g)
        self.x2c_act = nn.LeakyReLU(args.model_leakySlope_x)

        # init two conv layers
        for m in self.hyper1:
            m.reset_parameters()
        for m in self.hyper2:
            m.reset_parameters()

    def loss_mse(self, pred, label, weight, args):
        """
        mse loss, not used in the final version
        """
        rows, columns = label.shape
        users, phonePlacements, activity = args.users, args.phonePlacements, args.activities        

        if self.args.predict_user:
            loss_fn = nn.MSELoss(reduction='none')
            return (weight * loss_fn(pred, label)).mean()

        else:
            label_no_user = label[:, users:]
            loss_fn = nn.MSELoss(reduction='none')
            return (weight[:, users:]* loss_fn(pred, label_no_user)).mean()

    def loss_bce(self, pred, label, weight, args):
        """
        classification loss, multi-label problem
        """
        # loss_fn1 = nn.BCELoss()
        # loss_fn2 = nn.MSELoss()
        rows, columns = label.shape
        users, phonePlacements, activity = args.users, args.phonePlacements, args.activities
        

        if self.args.predict_user:
            loss_fn = nn.BCEWithLogitsLoss(reduction=self.args.reduction, weight=weight)
            return loss_fn(pred, label)

        else:
            label_no_user = label[:, users:]
            loss_fn = nn.BCEWithLogitsLoss(reduction=self.args.reduction, weight=weight[:, users:])
            return loss_fn(pred, label_no_user)

    def loss_contrast(self, g, args):
        """
        contrastive loss, contains homo loss and hetero loss, combine with linear weights
        """
        # last dim in pp is synthetic and all 0 in extrasensory, not joining computation
        g[1] = g[1][:-1, :]
        return args.lambda1 * self.loss_homo(g, args) - args.lambda2 * self.loss_hetero(g, args)

    def loss_homo(self, g, args):
        """
        homo loss
        """
        loss = [1. - torch.nanmean(pss(g[i], zero_diagonal=True)) for i in range(3)]
        return sum(loss) / 3. 

    def loss_hetero(self, g, args):
        """
        hetero loss
        """
        loss = [1. - torch.nanmean(pss(g[i], torch.cat([g[j] for j in range(3) if j!=i], dim=0))) for i in range(3)]
        return sum(loss) / 3.

    def loss(self, pred, label, weight, g, args):
        """
        total loss is here
        """
        g = [
            g[:self.args.users, :], 
            g[self.args.users:self.args.users+self.args.phonePlacements, :], 
            g[self.args.users+self.args.phonePlacements:, :],
            ]
        return self.loss_bce(pred, label, weight, args) + self.loss_contrast(g, args)

    def forward(self, x, graphData):
        # prepare graph info
        g, hyperWeight, hyperAttr = graphData[:3]
        hyperIndex = graphData[3:]

        # layer 0
        g = [
            g[:self.args.users, :], 
            g[self.args.users:self.args.users+self.args.phonePlacements, :], 
            g[self.args.users+self.args.phonePlacements:, :],
            ]
        g = [layer(g[i]) for i, layer in enumerate(self.layer0_linear)]
        g = torch.cat(g, dim=0)
        g = self.layer0_act(g)

        # layer1 Conv, u_2, pp_2, a_4
        g = [layer(g, hyperIndex[i], hyperWeight, hyperAttr) for i,layer in enumerate(self.hyper1)]
        g = torch.sum(torch.stack(g, dim=0), dim=0)

        # layer1 act and dropout
        g = self.act1(g)
        # layer1 linear
        g = [
            g[:self.args.users, :], 
            g[self.args.users:self.args.users+self.args.phonePlacements, :], 
            g[self.args.users+self.args.phonePlacements:, :],
            ]
        g = [layer(g[i]) for i, layer in enumerate(self.layer1_linear)]
        g = torch.cat(g, dim=0)
        g = self.layer1_act(g)

        # layer2 conv
        g = [layer(g, hyperIndex[i], hyperWeight, hyperAttr) for i,layer in enumerate(self.hyper2)]
        g = torch.sum(torch.stack(g, dim=0), dim=0)

        # layer2 act and dropout
        g = self.act2(g)
        
        # map matrices to same dim
        g = [
            g[:self.args.users, :], 
            g[self.args.users:self.args.users+self.args.phonePlacements, :], 
            g[self.args.users+self.args.phonePlacements:, :],
            ]
        new_g = [self.g2c_act(layer(g[i])) for i, layer in enumerate(self.g2c)]

        x = [self.x2c_act(layer(x)) for i, layer in enumerate(self.x2c)]

        # matrix multiplication for final prediction
        result = [torch.mm(x[i], new_g[i].T) for i in range(3)]
        # not predicting user
        result = torch.cat(result[1:], axis=1)
        return result, torch.cat(g, dim=0)

