import numpy as np
import math
import h5py
from sklearn.metrics import confusion_matrix


class ConfusionMatrix:
    def __init__(self, n_class, class_names):
        self.CM = np.zeros((n_class, n_class))
        self.n_class = n_class
        self.class_names = class_names

    def clear(self):
        self.CM = np.zeros((self.n_class, self.n_class))

    def add_batch(self, gt, pred):
        self.CM += confusion_matrix(gt, pred, labels=list(range(self.n_class)))

    def overall_accuracy(self):  # percentage of correct classification
        return 100 * self.CM.trace() / self.CM.sum()

    def class_IoU(self, show=0):
        ious = np.full(self.n_class, 0.)
        for i_class in range(self.n_class):
            ious[i_class] = self.CM[i_class, i_class] / \
                            (-self.CM[i_class, i_class] \
                             + self.CM[i_class, :].sum()
                             + self.CM[:, i_class].sum())
        if show:
            print('  |  '.join('{} : {:3.2f}%'.format(name, 100 * iou) for name, iou in zip(self.class_names, ious)))
        # do not count classes that are not present in the dataset in the mean IoU
        return 100*ious[0],100*ious[1],100*ious[2],100*ious[3],100*ious[4],100*ious[5],100 * np.nansum(ious) / (np.logical_not(np.isnan(ious))).sum()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True) # Return (values, indices) : The k largest elements  of the given input tensor along a given dimension. ZZC
    pred = pred.t()  # make symmetric positive definite

    correct = pred.eq(target.view(1, -1).expand_as(pred))  # Computes element-wise equality
    # Return a 1 at each location where comparison is true

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def adjust_learning_rate(optimizer, global_counter, batch_size, base_lr):
    lr = max(base_lr * (0.5 ** (global_counter*batch_size // 300000)), 1e-5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_bn_decay(epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    bn_momentum = 0.5 * (0.5 ** (epoch // 10))
    bn_decay = np.minimum(1-bn_momentum, 0.99)
    return bn_decay

def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx

def loadDataFile(filename):
    print('Start read h5 data...')
    f = h5py.File(filename)
    print('Read h5 data done!')
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]