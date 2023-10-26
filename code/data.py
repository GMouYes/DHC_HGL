import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class myDataset(Dataset):
    """ dataset reader
    """

    def __init__(self, args, dataType):
        super(myDataset, self).__init__()

        self.x = torch.tensor(np.load(args.dataPath + dataType + '/' + args.xPath + '.npy'), dtype=torch.float)
        self.y = torch.tensor(np.load(args.dataPath + dataType + '/' + args.yPath + '.npy'), dtype=torch.float)
        self.weight = torch.tensor(np.load(args.dataPath + dataType + '/' + args.weightPath + '.npy'), dtype=torch.float)
        self.g = torch.tensor(np.load(args.dataPath + 'train/' + args.nodePath + '.npy'), dtype=torch.float)
        self.hyperIndex = [torch.tensor(np.load(args.dataPath + 'train/' + args.hyperIndexPath + '_{}.npy'.format(i)), dtype=torch.long) for i in ['u_pp_a','u_pp','u_a']]
        self.hyperWeight = torch.tensor(np.load(args.dataPath + 'train/' + args.hyperWeightPath + '.npy'), dtype=torch.float)
        self.hyperAttr = torch.tensor(np.load(args.dataPath + 'train/' + args.hyperAttrPath + '.npy'), dtype=torch.float)

        # print(self.x.shape, self.y.shape, self.g.shape, self.hyperIndex.shape, self.hyperWeight.shape, self.hyperAttr.shape)
    
    def getGraph(self):
        return [self.g, self.hyperWeight, self.hyperAttr] + self.hyperIndex
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.weight[index], self.y[index]


def loadData(args, dataType):
    data = myDataset(args, dataType)
    print("{} size: {}".format(dataType, len(data.y)))

    shuffle = (dataType == "train")
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)
    return loader
