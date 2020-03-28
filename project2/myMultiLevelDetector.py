import argparse
import os
import time

import numpy as np
from torch import nn
import torch.optim as optim

from myMultiLevelData import get_train_test_set
from myMultiLevelNN import Net, resnet18, resnet34, resnet50, resnet101, resnet152, GoogLeNet
from myPredict import predict, test

import torch

torch.set_default_tensor_type(torch.FloatTensor)


def subtrain(img, device, landmark, optimizer, model, pts_criterion, retain=True):
    # ground truth
    input_img = img.to(device)
    target_pts = landmark.to(device)

    # clear the gradients of all optimized variables
    optimizer.zero_grad()

    # get output
    output_pts = model(input_img)



    return target_pts, output_pts


def subtest(valid_img, device, landmark, model, pts_criterion):
    input_img = valid_img.to(device)
    target_pts = landmark.to(device)

    output_pts = model(input_img)

    return target_pts, output_pts


def train(args, train_loader, valid_loader, model, criterion, optimizer, device, scheduler=None, cuda=False):
    loader_order = ['u.pt', 'd.pt', 's.pt']
    er = args.effective_ratio
    # save model
    if args.save_model:
        if not os.path.exists(args.save_directory):
            os.makedirs(args.save_directory)
    # load checkpoint
    if args.checkpoint != '':
        for i in range(len(model)):
            checkpoint = torch.load('{}/{}'.format(args.checkpoint, loader_order[i]), map_location={'cuda:0': 'cuda' if cuda else 'cpu'})
            model[i].load_state_dict(checkpoint['model_state_dict'])
            optimizer[i].load_state_dict(checkpoint['optimizer_state_dict'])
            print('Training from checkpoint: %s/%s' % (args.checkpoint, loader_order[i]))

    epoch = args.epochs
    pts_criterion = criterion

    l1_train_losses = []
    l2_train_losses = []
    l1_valid_losses = []
    l2_valid_losses = []
    time_elapse1 = []
    time_elapse2 = []

    for epoch_id in range(1, epoch + 1):
        ######################
        # training the model #
        ######################
        for m in model:
            m.train()
        start = time.perf_counter()
        for batch_idx, batch in enumerate(train_loader):
            img = batch['image']
            # Split face information into Eye part, noise part and mouth part
            landmarku = batch['landmarku']
            landmarkm = batch['landmarkd']
            landmarkd = batch['landmarks']
            # Train separately

            l1_result = [
                subtrain(img, device, landmarku, optimizer[0], model[0], pts_criterion),
                subtrain(img, device, landmarkm, optimizer[1], model[1], pts_criterion),
                subtrain(img, device, landmarkd, optimizer[2], model[2], pts_criterion, retain=False)
            ]
            #landmark = torch.cat([i[1].transpose(1, 0) for i in l1_result], dim=0)
            #l1_list = [[l1[:l] for l in range(l1.shape[1])] for l1 in l1_result]

            # do net 1 BP automatically
            loss1 = pts_criterion(l1_result[0][1], l1_result[0][0])
            loss1.backward(retain_graph=True)
            optimizer[0].step()

            # do net 2 BP automatically
            loss2 = pts_criterion(l1_result[1][1], l1_result[1][0])
            loss2.backward(retain_graph=True)
            optimizer[1].step()

            l1_result[2][1][:, :12] = (l1_result[2][1][:, :12]*(1-er)+l1_result[0][1][:, :12]*er)
            l1_result[2][1][:, 12:16] = (l1_result[2][1][:, 12:16]*(1-2*er)+l1_result[0][1][:, 12:16]*er+l1_result[1][1][:, :4]*er)
            l1_result[2][1][:, 16:] = (l1_result[2][1][:, 16:]*(1-er)+l1_result[1][1][:, 4:]*er)
            # do net 2 BP automatically
            loss3 = pts_criterion(l1_result[2][1], l1_result[2][0])
            loss3.backward()
            optimizer[2].step()

            if batch_idx % args.log_interval == 0:
                l1_train_losses.append([i[0] for i in l1_result])
                print('L1 Train Epoch: {} [{}/{} ({:.0f}%)]\t pts_loss: {:.6f} {:.6f} {:.6f}'.format(
                    epoch_id,
                    batch_idx * len(img),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss1,
                    loss2,
                    loss3
                )
                )


        if scheduler:  # Finetune with learning rate scheduler
            for sch in scheduler:
                sch.step()
        elapsed = time.perf_counter() - start
        print("Trained elapsed: %.5f" % elapsed)
        time_elapse1.append(elapsed)
        ######################
        # validate the model #
        ######################
        l1_valid_mean_pts_loss = 0.0

        for m in model:
            m.eval()  # prep model for evaluation
        start = time.perf_counter()
        with torch.no_grad():
            valid_batch_cnt = 0
            mean_loss1 = mean_loss2 = mean_loss3 = 0
            for valid_batch_idx, batch in enumerate(valid_loader):
                valid_batch_cnt += 1
                valid_img = batch['image']
                # Split face information into Eye part, noise part and mouth part
                landmarku = batch['landmarku']
                landmarkm = batch['landmarkd']
                landmarkd = batch['landmarks']

                l1_result = [
                    subtest(valid_img, device, landmarku, model[0], pts_criterion),
                    subtest(valid_img, device, landmarkm, model[1], pts_criterion),
                    subtest(valid_img, device, landmarkd, model[2], pts_criterion)
                ]
                # net 1
                loss1 = pts_criterion(l1_result[0][1], l1_result[0][0])

                # net 2
                loss2 = pts_criterion(l1_result[1][1], l1_result[1][0])

                # net 3
                l1_result[2][1][:, :12] = (l1_result[2][1][:, :12] * (1-er) + l1_result[0][1][:, :12] * er)
                l1_result[2][1][:, 12:16] = (l1_result[2][1][:, 12:16] * (1-2*er) + l1_result[0][1][:, 12:16] * er + l1_result[1][1][:, :4] * er)
                l1_result[2][1][:, 16:] = (l1_result[2][1][:, 16:] * (1-er) + l1_result[1][1][:, 4:] * er)
                loss3 = pts_criterion(l1_result[2][1], l1_result[2][0])


                mean_loss1 += loss1
                mean_loss2 += loss2
                mean_loss3 += loss3
            l1_valid_mean_pts_loss = [float(i/valid_batch_cnt*1.0) for i in (mean_loss1, mean_loss2, mean_loss3)]
            print('Valid L1: pts_loss:', l1_valid_mean_pts_loss)

            l1_valid_losses.append(l1_valid_mean_pts_loss)
        elapsed = time.perf_counter() - start
        print("Evaluation elapsed: %.5f" % elapsed)
        print('====================================================')
        time_elapse2.append(elapsed)
        # save model
        if args.save_model and epoch_id % args.save_interval == 0:
            f = os.path.join(args.save_directory, 'detector_epoch' + '_' + str(epoch_id))
            if not os.path.exists(f):
                os.makedirs(f)
            for i in range(len(model)):
                saved_model_name = os.path.join(f, loader_order[i])
                torch.save({'model_state_dict': model[i].state_dict(),
                            'optimizer_state_dict': optimizer[i].state_dict()
                            }, saved_model_name)
    return l1_train_losses, l1_valid_losses


def main_test():
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--alg', type=str, default='SGD',
                        help='select optimzer SGD, adam, or other')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-interval', type=int, default=20,
                        help='after # of epoch, save the current Model')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the current Model')
    parser.add_argument('--save-directory', type=str, default='trained_models',
                        help='learnt models are saving here')
    parser.add_argument('--phase', type=str, default='Train',  # Train/train, Predict/predict, Finetune/finetune
                        help='training, predicting or finetuning')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='continuing the training from specified checkpoint')
    parser.add_argument('--net', type=str, default='',
                        help='DefaultNet, ResNet***[18,34,50,101,152], MobileNet or GoogLeNet')
    parser.add_argument('--roi', type=float, default=0.25,
                        help='expand original face grid by ratio')
    parser.add_argument('--angle', type=float, default=10,
                        help='max (10) angle range to rotate original image on both side')
    parser.add_argument('--num-class', type=int, default=42,
                        help='default number of class 42')
    parser.add_argument('--scheduler', type=str, default='',
                        help='scheduler selection for fine tune phrase')
    parser.add_argument('--loss', type=str, default='L2',
                        help='loss function')
    parser.add_argument('--finetune-model', type=int, default=0,
                        help='select 1 model to finetune')
    parser.add_argument('--effective-ratio', type=float, default=0.2,
                        help='Given the effiective ratio of the three model')
    parser.add_argument('--layer-lockdown', type=str, default='0:14',
                        help='Freeze the specific layer')
    args = parser.parse_args()
    print(args)
    ###################################################################################
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")  # cuda:0
    # For multi GPUs, nothing need to change here
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    print('===> Loading Datasets')
    train_set, test_set = get_train_test_set(args.net, args.roi, args.angle)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size)
    ####################################################################
    print('===> Building Model')
    model = []
    model.append(Net(num_classes=16).to(device))
    model.append(Net(num_classes=9).to(device))
    model.append(Net(num_classes=21).to(device))
    ####################################################################
    if args.loss == 'L1':
        criterion_pts = nn.L1Loss()
    elif args.loss == 'SL1':
        criterion_pts = nn.SmoothL1Loss()
    else:
        criterion_pts = nn.MSELoss()
    ####################################################################
    if args.phase == 'Finetune':
        start, end = (args.layer_lockdown.split(':'))
        for m in range(len(model)):
            if m == args.finetune_model:
                for param in list(model[m].parameters())[start:end]:
                    param.requires_grad = False
            else:
                for param in list(model[m].parameters()):
                    param.requires_grad = False
    if args.alg == 'SGD':
        optimizer = [optim.SGD(m.parameters(), lr=args.lr, momentum=args.momentum) for m in model]
    elif args.alg == 'adam' or args.alg == 'Adam':
        optimizer = [optim.Adam(m.parameters(), lr=args.lr) for m in model]
    else:
        optimizer = [optim.Adam(m.parameters(), lr=args.lr) for m in model]
    ####################################################################
    if args.scheduler == 'StepLR100':
        scheduler = [torch.optim.lr_scheduler.StepLR(o, step_size=100) for o in optimizer]
    else:
        scheduler = None
    ####################################################################
    if args.phase == 'Train' or args.phase == 'train':
        print('===> Start Training')
        l1_train_losses, l1_valid_losses= \
            train(args, train_loader, valid_loader, model, criterion_pts, optimizer, device, cuda=use_cuda)
        with open(args.save_directory + 'train_result.txt', 'w+') as f:
            f.write(' '.join(l1_train_losses) + '\n')
            f.write(' '.join(l1_valid_losses) + '\n')
        print('====================================================')
    elif args.phase == 'Test' or args.phase == 'test':
        print('===> Test')
        test(args, model, valid_loader, output_file='output.txt')
        print('====================================================')
    elif args.phase == 'Finetune' or args.phase == 'finetune':
        print('===> Finetune')
        l1_train_losses, l1_valid_losses= \
            train(args, train_loader, valid_loader, model, criterion_pts, optimizer, device, scheduler=scheduler, cuda=use_cuda)
        with open(args.save_directory + 'train_result.txt', 'w+') as f:
            f.write(' '.join(l1_train_losses) + '\n')
            f.write(' '.join(l1_valid_losses) + '\n')
        print('====================================================')
    elif args.phase == 'Predict' or args.phase == 'predict':
        print('===> Predict')
        predict(args, model, valid_loader)
        print('====================================================')


if __name__ == '__main__':
    print(main_test())
