import argparse
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
from pprint import pprint

from model import HHGNN_hetero
from model import save_model
from util import *
from data import loadData
# import warnings filter
from warnings import simplefilter
from torch.nn import utils
simplefilter(action='ignore', category=UserWarning)


class Trainer(object):
    """
    Trainer object
    """

    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.device = 'cuda:0' if torch.cuda.is_available() and args.use_cuda else 'cpu'
        self.bestResult = np.inf

    def train(self, network, train_data, dev_data=None):
        """
        training process
        :param network: model to train on
        :param train_data: dataloader containing train split
        :param dev_data: dataloader containing dev split
        :return: boolean, placeholder indicating training success
        """
        network = network.to(self.device)
        train_loss, valid_loss = [], []
        validator = Tester(self.args)
        self.optimizer = torch.optim.RAdam(network.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.args.gamma)
        for epoch in range(1, self.args.epoch + 1):
            network.train()
            train_epoch_loss = self._train_step(train_data, network, epoch=epoch)
            train_loss.append(train_epoch_loss)
            print('[Trainer] loss: {}'.format(train_epoch_loss))

            # validation
            test_epoch_loss = validator.test(network, dev_data, epoch=epoch)
            valid_loss.append(test_epoch_loss)
            if self.best_eval_result(test_epoch_loss):
                save_model(network, self.args)
            self.scheduler.step()
        self.plot_loss(train_loss, valid_loss)

        return True

    def plot_loss(self, train_loss, valid_loss):
        """
        helper method to plot and check if the loss converges
        """

        plt.figure()
        ax = plt.subplot(121)
        ax.set_title('train loss')
        ax.plot(train_loss, 'r-')

        ax = plt.subplot(122)
        ax.set_title('validation loss')
        ax.plot(valid_loss, 'b-')

        plt.savefig('{}/{}_{}'.format(self.args.outputPath, self.args.lr, self.args.lossPath))
        plt.close()

        return True

    def _train_step(self, data_iterator, network, **kwargs):
        """
        Training process in one epoch.
        """
        # train_acc = 0.
        loss_record = 0.
        graphData = data_iterator.dataset.getGraph()
        graphData = [item.to(self.device) for item in graphData]
        for data in tqdm(data_iterator, desc="Train epoch {}".format(kwargs["epoch"])):
            x = data[0].to(self.device)
            # x = [item.to(self.device) for item in data[:1]+graphData]
            label = data[2].to(self.device)
            weight = data[1].to(self.device)

            self.optimizer.zero_grad()
            pred, g = network(x, graphData)
            loss = network.loss(pred, label, weight, g, self.args)
            loss_record += loss.item()
            loss.backward()
            utils.clip_grad_norm_(network.parameters(), self.args.clip_grad)
            self.optimizer.step()


        return loss_record / len(data_iterator.dataset)

    def best_eval_result(self, test_loss):
        """
        Check if the current epoch yields better validation results.

        :param test_loss: a floating number
        :return: bool, True means current results on dev set is the best.
        """

        if test_loss < self.bestResult:
            self.bestResult = test_loss
            return True
        return False


class Tester(object):
    """
    validating on the dev set
    """

    def __init__(self, args):
        super(Tester, self).__init__()
        self.args = args
        self.device = 'cuda:0' if torch.cuda.is_available() and args.use_cuda else 'cpu'

    def test(self, network, dev_data, **kwargs):
        network = network.to(self.device)
        network.eval()

        # test_acc = 0.0
        valid_loss = 0.0
        graphData = dev_data.dataset.getGraph()
        graphData = [item.to(self.device) for item in graphData]

        for data in tqdm(dev_data, desc="Test epoch {}".format(kwargs["epoch"])):
            x = data[0].to(self.device)
            label = data[2].to(self.device)
            weight = data[1].to(self.device)

            with torch.no_grad():
                pred, g = network(x, graphData)
                loss = network.loss(pred, label, weight, g, self.args)
                valid_loss += loss.item()

        valid_loss /= len(dev_data.dataset)
        print("[Tester] Loss: {}".format(valid_loss))

        return valid_loss


class Predictor(object):
    """
    Final inference on the testing data
    """

    def __init__(self, args):
        super(Predictor, self).__init__()
        self.args = args
        self.device = 'cuda:0' if torch.cuda.is_available() and args.use_cuda else 'cpu'

    def predict(self, network, test_data):
        network = network.to(self.device)
        network.eval()

        pred_list, truthList, mask_list = [], [], []
        graphData = test_data.dataset.getGraph()
        graphData = [item.to(self.device) for item in graphData]
        test_acc = 0.0

        for data in tqdm(test_data, desc="Final Infer:"):
            x = data[0].to(self.device)
            label = data[2].to(self.device)
            weight = data[1].to(self.device)

            with torch.no_grad():
                pred, g = network(x, graphData)

            pred_list.append(pred.detach().cpu())
            truthList.append(label.detach().cpu())
            mask_list.append(weight.detach().cpu())

        pred_list = torch.cat(pred_list, axis=0)
        truthList = torch.cat(truthList, axis=0)
        mask_list = torch.cat(mask_list, axis=0)
        g = g.detach().cpu()

        if self.args.mask:
            test_result = metric(truthList, pred_list, self.args, mask_list)
        else:
            test_result = metric(truthList, pred_list, self.args)

        metricNames = ['BA', 'MCC', 'MacroF1']
        for names, result in zip(metricNames, test_result):
            print(names)
            startInd = 0
            print("[Final tester]: {:.4f}".format(np.nanmean(result)))

            print("Total: {:.4f}".format(np.nanmean(result[startInd:])))
            print("PP: {:.4f}".format(np.nanmean(result[startInd:startInd+self.args.phonePlacements])))
            print("Activity: {:.4f}".format(np.nanmean(result[startInd+self.args.phonePlacements:])))
            print(result)
            print("")
        return pred_list, g

def model(args):
    """
    1. print params
    2. create output path
    3. fix seeds
    4. train and test data, evaluate and save all results

    :param args: argparse settings for running the model
    """

    pprint(args)

    if not os.path.isdir(args.outputPath):
        os.mkdir(args.outputPath)

    seed_all(args.seed)
    trainData, validData, testData = [loadData(args, dataType) for dataType in ['train', 'valid', 'test']]
    model = HHGNN_hetero(args)

    trainer = Trainer(args)
    trainer.train(model, trainData, validData)

    model.load_state_dict(torch.load(args.outputPath + '/' + args.modelPath))
    predictor = Predictor(args)
    pred, g = predictor.predict(model, testData)
    np.save(args.outputPath + '/' + args.resultPath, pred)
    np.save(args.outputPath + '/' + 'nodes_' + args.resultPath, g)
    return


def main():
    """
    starting point
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--expName', type=str, default='debug')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--reduction', type=str, default='sum')
    parser.add_argument('--clip_grad', type=float, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--hetero', action='store_true')
    parser.add_argument('--smoothing', type=float, default=0.1)

    parser.add_argument('--users', type=int, required=True)
    parser.add_argument('--phonePlacements', type=int, required=True)
    parser.add_argument('--activities', type=int, required=True)
    parser.add_argument('--predict_user', action='store_true')
    parser.add_argument('--mask', action='store_true')
    parser.add_argument('--lambda1', type=float, default=0.1)
    parser.add_argument('--lambda2', type=float, default=0.1)
    
    parser.add_argument('--dataPath', type=str, default='./data/')
    parser.add_argument('--outputPath', type=str, default='./output/')
    parser.add_argument('--xPath', type=str, default='')
    parser.add_argument('--yPath', type=str, default='')
    parser.add_argument('--weightPath', type=str, default='')
    parser.add_argument('--nodePath', type=str, default='')
    parser.add_argument('--hyperIndexPath', type=str, default='')
    parser.add_argument('--hyperWeightPath', type=str, default='')
    parser.add_argument('--hyperAttrPath', type=str, default='')
    parser.add_argument('--modelPath', type=str, default='./output/')
    parser.add_argument('--resultPath', type=str, default='./output/')
    parser.add_argument('--lossPath', type=str, default='loss.pdf')

    parser.add_argument('--model_dropout1', type=float, default=0.05)
    parser.add_argument('--model_dropout2', type=float, default=0.05)
    parser.add_argument('--model_commonDim', type=int, default=128)
    parser.add_argument('--model_leakySlope_g', type=float, default=0.2)
    parser.add_argument('--model_leakySlope_x', type=float, default=0.2)
    parser.add_argument('--model_finalLeakySlope', type=float, default=0.2)
    parser.add_argument('--XLeakySlope', type=float, default=0.2)
    parser.add_argument('--xMidDim', type=int, default=256)

    parser.add_argument('--hgcn_l1_before_leakySlope', type=float, default=0.2)
    parser.add_argument('--hgcn_l1_in_channels', type=int, default=-1)
    parser.add_argument('--hgcn_l1_out_channels', type=int, default=128)
    parser.add_argument('--hgcn_l1_use_attention', action='store_true')
    parser.add_argument('--hgcn_l1_heads', type=int, default=128)
    parser.add_argument('--hgcn_l1_concat', action='store_true')
    parser.add_argument('--hgcn_l1_negative_slope', type=float, default=0.2)
    parser.add_argument('--hgcn_l1_dropout', type=float, default=0.05)
    parser.add_argument('--hgcn_l1_bias', action='store_true')
    parser.add_argument('--hgcn_l1_after_leakySlope', type=float, default=0.2)

    parser.add_argument('--hgcn_l2_before_leakySlope', type=float, default=0.2)
    parser.add_argument('--hgcn_l2_in_channels', type=int, default=-1)
    parser.add_argument('--hgcn_l2_out_channels', type=int, default=128)
    parser.add_argument('--hgcn_l2_use_attention', action='store_true')
    parser.add_argument('--hgcn_l2_heads', type=int, default=128)
    parser.add_argument('--hgcn_l2_concat', action='store_true')
    parser.add_argument('--hgcn_l2_negative_slope', type=float, default=0.2)
    parser.add_argument('--hgcn_l2_dropout', type=float, default=0.05)
    parser.add_argument('--hgcn_l2_bias', action='store_true')
    parser.add_argument('--hgcn_l2_after_leakySlope', type=float, default=0.2)

    args = parser.parse_args()
    model(args)
    return

if __name__ == '__main__':
    main()
