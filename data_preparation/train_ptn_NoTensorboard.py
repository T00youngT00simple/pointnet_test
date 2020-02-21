from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import argparse
from datetime import datetime
import h5py

# from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
from torch.autograd import Variable

from model.pointnet import PointNet
from utils.train_utils import ConfusionMatrix, adjust_learning_rate, shuffle_data, getDataFiles

def main():
    parser = argparse.ArgumentParser(description='Voxelnet for semantic')
    parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')   # default=0.001(good)
    parser.add_argument('--epochs', default=2, help='epochs')   # default=100, 50, 30
    parser.add_argument('--batchsize', default=4, help='epochs')   # default=32
    parser.add_argument('--weight_file', default='', help='weights to load')
    # log_ptn/train/Area_2_2019-09-11-11-43-48/checkpoint/checkpoint_0_max_mIoU_test_25.17065278824228.pth.tar
    parser.add_argument('--test_area', type=int, default=2, help='Which area to use for test, option: 1-2 [default: 2]')
    parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')

    args = parser.parse_args()
    NUM_POINT = args.num_point
    BATCH_SIZE = args.batchsize
    lr = args.lr
    ALL_FILES = getDataFiles('indoor3d_sem_seg_hdf5_data/all_files.txt')  # .h5 file routes
    room_filelist = [line.rstrip() for line in open('indoor3d_sem_seg_hdf5_data/room_filelist.txt')]

    # Load ALL data into a big data_batch & a big label_batch
    data_batch_list = []
    label_batch_list = []
    print(ALL_FILES)
    for h5_filename in ALL_FILES:
        h5_dir = os.path.join('/home/chenkun/pointnet_pytorch-master/indoor3d_sem_seg_hdf5_data',h5_filename)
        f = h5py.File(h5_dir)
        data_batch = f['data'][:]
        label_batch = f['label'][:]
        data_batch_list.append(data_batch)
        label_batch_list.append(label_batch)
    data_batches = np.concatenate(data_batch_list, 0)
    label_batches = np.concatenate(label_batch_list, 0)
    print(data_batches.shape)
    print(label_batches.shape)

    test_area = 'Area_' + str(args.test_area)
    train_idxs = []
    test_idxs = []
    for i, room_name in enumerate(room_filelist):
        if test_area in room_name:
            test_idxs.append(i)
        else:
            train_idxs.append(i)

    train_data = data_batches[train_idxs, ...]
    train_label = label_batches[train_idxs].astype(np.int64)
    # test_data = data_batches[test_idxs, ...]      # ZZC
    # test_label = label_batches[test_idxs].astype(np.int64)  # ZZC

    test_data = train_data    # ZZC
    test_label = train_label    # ZZC

    print(train_data.shape, train_label.shape)
    print(test_data.shape, test_label.shape)


    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_dir = os.path.join('log_ptn/train', test_area + '_' + time_string)

    if not os.path.exists(log_dir): os.makedirs(log_dir)

    checkpoint_dir = os.path.join(log_dir, 'checkpoint')
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)


    start_epoch = 0
    epochs = args.epochs

    model = get_model()
    model.cuda()
    # print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr)

    # class_names = ["ground", "vegetation", "building", "clutter"]    # ZZC
    class_names = ["T2T", "B2B", "BH", "BL", "V2V", "OT"]

    # Add weights to the loss function
    # weightsTrain = [0.04, 0.20, 0.12, 0.64]  # default
    # weightsTrain = [0.25, 0.25, 0.25, 0.25]
    # weightsTrain = [0.20, 0.50, 0.30, 0.50]
    weightsTrain = [0.2, 0.4, 0.6, 1.00, 1.00, 1.00]
    class_weights_Train = torch.FloatTensor(weightsTrain).cuda()
    criterionTrain = nn.CrossEntropyLoss(weight=class_weights_Train,
                                         size_average=True).cuda()
     # True: loss is averaged over each loss element in batch
    weightsVal = [0.2, 0.4, 0.6, 1.00, 1.00, 1.00]    # default  [0.08, 0.37, 0.15, 0.40]
    class_weights_Val = torch.FloatTensor(weightsVal).cuda()
    criterionVal = nn.CrossEntropyLoss(weight=class_weights_Val,
                                         size_average=True).cuda()

    if args.weight_file != '':
        pre_trained_model = torch.load(args.weight_file)
        start_epoch = pre_trained_model['epoch']
        model_state = model.state_dict()
        model_state.update(pre_trained_model['state_dict'])
        model.load_state_dict(model_state)


    #  #####################################################
    #    Start training
    #  #####################################################
    global_counter = 0
    max_mIoU_test = 0.0

    for epoch in range(start_epoch, epochs):
        learn_rate_now = adjust_learning_rate(optimizer, global_counter, BATCH_SIZE, lr)  # Seems not changing, ZZC

        iter_loss = 0.0  # Initialisation: loss for one epoch
        iterations = 0

        cm = ConfusionMatrix(6, class_names=class_names)
        cm.clear()

        model.train()

        train_data_shuffled, train_label_shuffled, _ = shuffle_data(train_data[:, 0:NUM_POINT, :], train_label)
        file_size = train_data_shuffled.shape[0]  # total number of training batches
        num_batches = file_size // BATCH_SIZE  # number of iterations in one epoch
        print('\nnum_batches(training):\t',num_batches)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE

            feature = train_data_shuffled[start_idx:end_idx, :, :]
            label = train_label_shuffled[start_idx:end_idx]
            # print('Here')
            # print(feature.shape)
            # print(label.shape)

            # feature[:, :, 0:2] = 0.0
            # feature[:, :, 6:9] = 0.0
            # print(feature.shape)

            # print(feature[0, 0, 0])
            # print(feature[0, 0, 1])
            # print(feature[0, 0, 2])
            # print(feature[0, 0, 3])
            # print(feature[0, 0, 4])
            # print(feature[0, 0, 5])
            # print(feature[0, 0, 6])
            # print(feature[0, 0, 7])
            # print(feature[0, 0, 8])

            #

            feature = np.expand_dims(feature, axis=1)
            input = Variable(torch.from_numpy(feature).cuda(), requires_grad=True)
            # print(input.size())

            input = torch.transpose(input, 3, 1)   # ? ZZC
            # print(input.size())

            target = Variable(torch.from_numpy(label).cuda(), requires_grad=False)
            # print(target.size())

            target = target.view(-1,)
            # print(target.size())

            output = model(input)
            output_reshaped = output.permute(0, 3, 2, 1).contiguous().view(-1, 6)

            # exit()  # for check, ZZC
            _, pred = torch.max(output.data, 1)
            pred = pred.view(-1,)
            cm.add_batch(target.cpu().numpy(), pred.cpu().numpy())  # detach()
            loss = criterionTrain(output_reshaped, target)
            iter_loss += loss.item()  # Accumulate the loss
            iterations +=1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_counter += 1

            if batch_idx%10==0:
                print('Epoch: [%3d][%3d]\t Loss: %.4f'%(epoch,batch_idx,loss))   # Print loss for one bath

        # Print training results for 1 epoch
        iou0,iou1,iou2,iou3,iou4,iou5,mIoU = cm.class_IoU()
        print('Epoch: [%3d]\t Train Loss: %.4f\t OA: %3.2f%%\t mIoU : %3.2f%%'%(epoch,iter_loss/iterations,cm.overall_accuracy(), mIoU))   # Print loss for the epoch
        print('T2T: %3.2f%%, B2B: %3.2f%%, BH: %3.2f%%, BL: %3.2f%%, V2V: %3.2f%%, OT: %3.2f%%'%(iou0,iou1,iou2,iou3,iou4,iou5))

        with open(os.path.join(log_dir, 'train_log.txt'), 'a') as f:
            f.write('Epoch: [%3d]\t Train Loss: %.4f\t OA: %3.2f%%\t mIoU : %3.2f%%\n'%(epoch,iter_loss/iterations,cm.overall_accuracy(), mIoU))
            f.write('T2T: %3.2f%%, B2B: %3.2f%%, BH: %3.2f%%, BL: %3.2f%%, V2V: %3.2f%%, OT: %3.2f%%\n\n'%(iou0,iou1,iou2,iou3,iou4,iou5))


        #  #####################################################
        #    Start validation
        #  #####################################################
        model.eval()
        iter_loss = 0.0  # Initialisation: loss for one epoch
        iterations = 0
        cm = ConfusionMatrix(6, class_names=class_names)   # ZZC
        cm.clear()

        file_size = test_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        print('num_batches(testing):\t',num_batches)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE
            feature = test_data[start_idx:end_idx, :, :]
            label = test_label[start_idx:end_idx]

            # feature[:, :, 0:2] = 0.0
            # feature[:, :, 6:9] = 0.0

            feature = np.expand_dims(feature, axis=1)
            input = Variable(torch.from_numpy(feature).cuda(), requires_grad=True)
            input = torch.transpose(input, 3, 1)   # ? ZZC
            target = Variable(torch.from_numpy(label).cuda(), requires_grad=False)
            target = target.view(-1,)
            output = model(input)
            output_reshaped = output.permute(0, 3, 2, 1).contiguous().view(-1, 6)

            _, pred = torch.max(output.data, 1)
            pred = pred.view(-1,)
            cm.add_batch(target.cpu().numpy(), pred.cpu().numpy())  # detach()

            loss = criterionVal(output_reshaped, target)
            iter_loss += loss.item()  # Accumulate the loss
            iterations +=1

        # Print validation results after 1 epoch
        iou0, iou1, iou2, iou3, iou4, iou5, mIoU = cm.class_IoU()
        print('Epoch: [%3d]\t Test Loss: %.4f\t OA: %3.2f%%\t mIoU : %3.2f%%'%(epoch,iter_loss/iterations,cm.overall_accuracy(), mIoU))   # Print loss for the epoch
        print('T2T: %3.2f%%, B2B: %3.2f%%, BH: %3.2f%%, BL: %3.2f%%, V2V: %3.2f%%, OT: %3.2f%%' % (iou0, iou1, iou2, iou3,iou4,iou5))

        with open(os.path.join(log_dir, 'test_log.txt'), 'a') as f:
            f.write('Epoch: [%3d]\t Test Loss: %.4f\t OA: %3.2f%%\t mIoU : %3.2f%%\n' % (epoch, iter_loss / iterations,cm.overall_accuracy(), mIoU))
            f.write('T2T: %3.2f%%, B2B: %3.2f%%, BH: %3.2f%%, BL: %3.2f%%, V2V: %3.2f%%, OT: %3.2f%%\n\n' % (iou0, iou1, iou2, iou3,iou4,iou5))

        # Check whether best model, -> Save model
        if (mIoU > max_mIoU_test or epoch == epochs - 1):
            max_mIoU_test = mIoU
            print('-> Best performance (test mIoU) achieved or This is final epoch.')
            print('Max_mIoU in testing: %3.2f%%\n'%(max_mIoU_test))
            torch.save(
                {'epoch': epoch + 1, 'args': args, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                os.path.join(checkpoint_dir, 'checkpoint_' + str(epoch) + '_max_mIoU_test_' + str(mIoU) + '.pth.tar') )



def get_model():
    model = PointNet()
    print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    return model

if __name__ == '__main__':
    main()