import argparse
import os
import time

from torch import nn
import torch.optim as optim

from myDataS3 import get_train_test_set_w_err
from myNNS3 import Net
from myPredict import predict, test
from myFPN import FPN101
import torch.nn.functional as F

import torch

# imports the torch_xla package
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
except ModuleNotFoundError:
    print('No TPUs scheme')

torch.set_default_tensor_type(torch.FloatTensor)


def train(args, train_loader, valid_loader, model, criterion1, criterion2, optimizer, device, scheduler=None, cuda=False):
    # save model
    if args.save_model:
        if not os.path.exists(args.save_directory):
            os.makedirs(args.save_directory)
    if args.checkpoint != '':
        model.load_state_dict(torch.load(args.checkpoint, map_location={'cuda:0': 'cuda' if cuda else 'cpu'}))
        print('Training from checkpoint: %s' % args.checkpoint)

    epoch = args.epochs

    train_losses = []
    valid_losses = []
    time_elapse1 = []
    time_elapse2 = []

    for epoch_id in range(1, epoch + 1):
        # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0
        ######################
        # training the model #
        ######################
        model.train()
        start = time.perf_counter()
        for batch_idx, batch in enumerate(train_loader):
            img = batch['image']
            landmark = batch['landmarks'].to(device)

            # ground truth
            input_img = img.to(device)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # get output
            output_m1, output_m2 = model(input_img)

            target_face = torch.FloatTensor([[1, 0] if i else [0, 1] for i in batch['face']]).to(device)
            output_face = F.softmax(output_m1, dim=1).to(device)

            # calculate face detection loss
            loss1 = criterion1(output_face, target_face)

            # construct new dataset for face key point output
            output_m2 = torch.index_select(output_m2, 0, torch.tensor(
                [1 for i in range(batch['face'].shape[0]) if batch['face'][i]]).to(device))
            target_pts = torch.index_select(landmark, 0, torch.tensor(
                [1 for i in range(batch['face'].shape[0]) if batch['face'][i]]).to(device))

            # calculate loss
            loss2 = criterion2(output_m2, target_pts)

            # calculate total loss
            loss = loss1 * img.shape[2] + loss2 * batch['face'].shape[0] / sum(batch['face'])

            # do BP automatically
            loss.backward()
            optimizer.step()

            # show log info
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t pts_loss: {:.6f} {:.6f} {:.6f}'.format(
                    epoch_id,
                    batch_idx * len(img),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss1.item(),
                    loss2.item(),
                    loss.item()
                )
                )
                train_losses.append((loss1.item(), loss2.item()))

        if scheduler:  # Finetune with learning rate scheduler
            scheduler.step()
        elapsed = time.perf_counter() - start
        print("Trained elapsed: %.5f" % elapsed)
        time_elapse1.append(elapsed)
        ######################
        # validate the model #
        ######################
        valid_mean_pts_loss = 0.0

        model.eval()  # prep model for evaluation
        start = time.perf_counter()
        with torch.no_grad():
            valid_batch_cnt = 0

            for valid_batch_idx, batch in enumerate(valid_loader):
                valid_batch_cnt += 1
                valid_img = batch['image']
                landmark = batch['landmarks'].to(device)

                input_img = valid_img.to(device)

                # get output
                output_m1, output_m2 = model(input_img)

                target_face = torch.FloatTensor([[1, 0] if i else [0, 1] for i in batch['face']]).to(device)
                output_face = F.softmax(output_m1, dim=1).to(device)

                # calculate face detection loss
                loss1 = criterion1(output_face, target_face)

                # construct new dataset for face key point output
                output_m2 = torch.index_select(output_m2, 0, torch.tensor(
                    [1 for i in range(batch['face'].shape[0]) if batch['face'][i]]).to(device))
                target_pts = torch.index_select(landmark, 0, torch.tensor(
                    [1 for i in range(batch['face'].shape[0]) if batch['face'][i]]).to(device))

                # calculate loss
                loss2 = criterion2(output_m2, target_pts)

                # calculate total loss
                loss = loss1 * img.shape[2] + loss2 * batch['face'].shape[0] / sum(batch['face'])

                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\t pts_loss: {:.6f} {:.6f} {:.6f}'.format(
                        epoch_id,
                        batch_idx * len(img),
                        len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss1.item(),
                        loss2.item(),
                        loss.item()
                    )
                    )

                valid_mean_pts_loss += loss.item()

            valid_mean_pts_loss /= valid_batch_cnt * 1.0
            print('Valid: pts_loss: {:.6f}'.format(
                valid_mean_pts_loss
            )
            )
            valid_losses.append(valid_mean_pts_loss)
        elapsed = time.perf_counter() - start
        print("Evaluation elapsed: %.5f" % elapsed)
        print('====================================================')
        time_elapse2.append(elapsed)
        # save model
        if args.save_model and epoch_id % args.save_interval == 0:
            saved_model_name = os.path.join(args.save_directory, 'detector_epoch' + '_' + str(epoch_id) + '.pt')
            torch.save(model.state_dict(), saved_model_name)
    return train_losses, valid_losses, time_elapse1, time_elapse1


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
    parser.add_argument('--alg', type=str, default='adam',
                        help='select optimzer SGD, adam, or other')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='momentum (default: 0.5)')
    parser.add_argument('--use-tpu', action='store_true', default=False,
                        help='enable cloud tpu training')
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
    parser.add_argument('--angle', type=float, default=30,
                        help='max (30) angle range to rotate original image on both side')
    parser.add_argument('--num-class', type=int, default=42,
                        help='default number of class 42')
    parser.add_argument('--scheduler', type=str, default='',
                        help='scheduler selection for fine tune phrase')
    parser.add_argument('--loss1', type=str, default='L2',
                        help='loss2 function')
    parser.add_argument('--loss2', type=str, default='L2',
                        help='loss2 function')
    parser.add_argument('--layer-lockdown', type=str, default='-3:-1',
                        help='Freeze the specific layer')
    parser.add_argument('--use-bn', action='store_true', default=False,
                        help='Add batch normalization to the first conv1_1')
    args = parser.parse_args()
    print(args)
    ###################################################################################
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")  # cuda:0
    # For multi GPUs, nothing need to change here
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    print('===> Loading Datasets')
    train_set, test_set = get_train_test_set_w_err(args.net, args.roi, args.angle)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size)
    ####################################################################
    print('===> Building Model')
    model = Net(args.use_bn).to(device)
    ####################################################################
    if args.loss1 == 'L2':
        criterion_pts1 = nn.MSELoss()
    elif args.loss1 == 'L1':
        criterion_pts1 = nn.L1Loss()
    elif args.loss1 == 'SL1':
        criterion_pts1 = nn.SmoothL1Loss()
    if args.loss2 == 'L2':
        criterion_pts2 = nn.MSELoss()
    elif args.loss2 == 'L1':
        criterion_pts2 = nn.L1Loss()
    elif args.loss2 == 'SL1':
        criterion_pts2 = nn.SmoothL1Loss()
    ####################################################################
    # Freeze all layer except ip3, if current mode is finetune
    if args.phase == 'Finetune':
        start, end = (args.layer_lockdown.split(':'))
        for param in list(model.parameters())[start:end]:
            param.requires_grad = False
    if args.alg == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.alg == 'adam' or args.alg == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ####################################################################
    # Add scheduler
    if args.scheduler == '':
        scheduler = None
    elif args.scheduler == 'StepLR100':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25)
    ####################################################################
    if args.phase == 'Train' or args.phase == 'train':
        print('===> Start Training')
        train_losses, valid_losses, time_elapse1, time_elapse2 = \
            train(args, train_loader, valid_loader, model, criterion_pts1, criterion_pts2, optimizer,
                device, scheduler=scheduler, cuda=use_cuda)
        with open(args.save_directory + 'train_result.txt', 'w+') as f:
            f.write(' '.join([str(i) for i in train_losses]) + '\n')
            f.write(' '.join([str(i) for i in valid_losses]) + '\n')
            f.write(' '.join(str(time_elapse1)) + '\n')
            f.write(' '.join(str(time_elapse2)) + '\n')
        print('====================================================')
    elif args.phase == 'Test' or args.phase == 'test':
        print('===> Test')
        test(args, model, valid_loader, output_file='output.txt', cuda=use_cuda)
        print('====================================================')
    elif args.phase == 'Finetune' or args.phase == 'finetune':
        print('===> Finetune')
        train_losses, valid_losses, time_elapse1, time_elapse2 = \
            train(args, train_loader, valid_loader, model, criterion_pts, optimizer, device, scheduler=scheduler,
                  cuda=use_cuda)
        with open(args.save_directory + 'train_result.txt', 'w+') as f:
            f.write(' '.join([str(i) for i in train_losses]) + '\n')
            f.write(' '.join([str(i) for i in valid_losses]) + '\n')
            f.write(' '.join(str(time_elapse1)) + '\n')
            f.write(' '.join(str(time_elapse2)) + '\n')
        print('====================================================')
    elif args.phase == 'Predict' or args.phase == 'predict':
        print('===> Predict')
        predict(args, model, valid_loader, cuda=use_cuda)
        print('====================================================')


if __name__ == '__main__':
    print(main_test())
