import time
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import balanced_accuracy_score as BA
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import f1_score as F1
import numpy as np
import random
import time
import itertools
import torch.nn.functional as F
import torch

def seed_all(seed):
    # seed everything
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    return True


def timing(func):
    def wrapper(args):
        print("Running function {}".format(func.__name__))
        t1 = time.time()
        res = func(args)
        t2 = time.time()
        period = t2 - t1
        print("{} took {} hour {} min {} sec".format(func.__name__, period // 3600, (period % 3600) // 60,
                                                     int(period % 60)))
        return res

    return wrapper

# rewrite this one to fit your need
def metric(label, pred, args, mask=None): 
    rows, columns = label.shape
    users, phonePlacements, activity = args.users, args.phonePlacements, args.activities

    label = label[:, users:]
    mask = mask[:, users:]
    users = 0
    
    target_pred = pred[:, users:]
    target_label = label[:, users:]
    target_pred = torch.sigmoid(target_pred)
    target_pred = (target_pred > 0.5).long()

    if mask is None:
        targetBA = [BA(target_label[:, i], target_pred[:, i]) for i in range(args.activities+args.phonePlacements)]
        targetMCC = [MCC(target_label[:, i], target_pred[:, i]) for i in range(args.activities+args.phonePlacements)]
        targetF1 = [F1(target_label[:, i], target_pred[:, i], average='macro') for i in range(args.activities+args.phonePlacements)]
    else:
        act_mask = mask[:, users:]
        targetBA, targetMCC, targetF1 = [], [], []
        for i in range(args.activities+args.phonePlacements):
            act_label = target_label[:, i]
            act_pred = target_pred[:, i]
            act_mask_tmp = act_mask[:, i] > 0

            act_target = [(item1, item2) for item1, item2, item3 in zip(act_label, act_pred, act_mask_tmp) if item3]
            act_label, act_pred = list(zip(*act_target))
            # act_label = [item1 for item1, item2 in zip(act_label, act_mask_tmp) if item2]
            # act_pred = [item1 for item1, item2 in zip(act_pred, act_mask_tmp) if item2]
            targetBA.append(BA(act_label, act_pred))
            targetMCC.append(MCC(act_label, act_pred))
            targetF1.append(F1(act_label, act_pred, average='macro'))

    resultBA = targetBA
    resultMCC = targetMCC
    resultMacroF1 = targetF1
    return np.array(resultBA), np.array(resultMCC), np.array(resultMacroF1)