import os

import torch
import torch.nn.functional as F

# 对于stage3， 唯一的不同在于，需要接收除了pts以外，还有：label与分类loss。


def predict(args, model1, model2, valid_loader, device, criterion, cuda=False):
    checkpoint1 = os.path.join(args.checkpoint, '1.pt')
    checkpoint2 = os.path.join(args.checkpoint, '2.pt')
    tl1 = torch.load(checkpoint1, map_location={'cuda:0': 'cuda' if cuda else 'cpu'})
    tl2 = torch.load(checkpoint2, map_location={'cuda:0': 'cuda' if cuda else 'cpu'})
    model1.load_state_dict(tl1)
    model2.load_state_dict(tl2)
    model1.eval()  # prep model for evaluation
    model2.eval()
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            # forward pass: compute predicted outputs by passing inputs to the model
            img = batch['image'].to(device)
            landmark = batch['landmarks']
            print('i: ', i)
            # generated
            output_m1 = model1(img)

            target_face = torch.FloatTensor([[1, 0] if i else [0, 1] for i in batch['face']]).to(device)
            output_face = F.softmax(output_m1, dim=1).to(device)

            # calculate face detection loss
            # loss1 = criterion1(output_face, target_face)
            output_sign = map(lambda x: x[0]-x[1], output_face)
            target_sign = map(lambda x: x[0]-x[1], target_face)
            result = [1 if x*y > 0 else -1 for x, y in zip(output_sign, target_sign)]
            accuracy = len([i for i in result if i > 0])/len(result)
            print("Accuracy: {:.6f}".format(accuracy))

            # construct new dataset for face key point output
            input_m2 = torch.index_select(img, 0, torch.tensor(
                [1 for i in range(batch['face'].shape[0]) if batch['face'][i]]).to(device))
            target_pts = torch.index_select(landmark, 0, torch.tensor(
                [1 for i in range(batch['face'].shape[0]) if batch['face'][i]]).to(device))


            # Model 2
            output_m2 = model2(input_m2)

            # calculate loss
            loss2 = criterion(output_m2, target_pts)
            print('Predict:  pts_loss: {:.6f}'.format(
                loss2.item()
            )
            )


def test(args, model1, model2, valid_loader, device, output_file='output.txt', cuda=False):
    checkpoint1 = os.path.join(args.checkpoint, '1.pt')
    checkpoint2 = os.path.join(args.checkpoint, '2.pt')
    model1.load_state_dict(torch.load(checkpoint1, map_location={'cuda:0': 'cuda' if cuda else 'cpu'}))
    model2.load_state_dict(torch.load(checkpoint2, map_location={'cuda:0': 'cuda' if cuda else 'cpu'}))
    model1.eval()  # prep model for evaluation
    model2.eval()
    with torch.no_grad():
        with open(output_file, 'w+') as f:
            for i, batch in enumerate(valid_loader):
                # forward pass: compute predicted outputs by passing inputs to the model
                img = batch['image']
                rect = batch['rect']
                path = batch['path']
                # generated
                output_m1 = model1(img)

                # construct new dataset for face key point output
                input_m2 = torch.index_select(img, 0, torch.tensor(
                    [1 for i in range(batch['face'].shape[0]) if batch['face'][i]]).to(device))

                # Model 2
                output_m2 = model2(input_m2)

                output_m1_s, output_m2_s = F.softmax(output_m1, dim=1).numpy(), output_m2.numpy()
                for r in range(len(output_m1_s)):
                    f.write(' '.join(str(i) for i in output_m1_s[r])+'\n')
                    f.write(' '.join(('%s %s' % (path[r], ' '.join([str(float(rect[i][r])) for i in range(4)])), ' '.join([str(i) for i in output_m2_s[r]])))+'\n')
