import argparse
import torch
import os
import shutil
import numpy as np
import criterion
from model import MetricLearner
from dataset import MetricData, SourceSampler

def get_args():
    parser = argparse.ArgumentParser(description='Face Occlusion Regression')
    # train
    parser.add_argument('--pretrain', type=str, default='/root/.torch/models/googlenet-1378be20.pth', help='pretrain googLeNet model paht')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs plans to train in total')
    parser.add_argument('--epoch_start', type=int, default=0, help='start epoch to count from')
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--ckpt', type=str, default='./ckpt', help='checkpoint folder')
    parser.add_argument('--resume', action='store_true', help='load previous best model and resume training')
    # annotation
    parser.add_argument('--anno', type=str, required=True, help='location of annotation file')
    parser.add_argument('--anno_test', type=str, required=True, help='location of test data annotation file')
    parser.add_argument('--img_folder', type=str, required=True, help='folder of image files in annotation file')
    parser.add_argument('--idx_file', type=str, required=True, help='idx file for every label class')
    # model hyperparameter
    parser.add_argument('--in_size', type=int, default=128, help='input tensor shape to put into model')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    device = torch.device('cuda:{}'.format(args.gpu_ids[0])) if args.gpu_ids else torch.device('cpu')
    data = MetricData(data_root=args.img_folder, anno_file=args.anno, idx_file=args.idx_file)
    dataset = torch.utils.data.DataLoader(data, batch_size=args.batch, sampler=SourceSampler(data, args.batch//2), drop_last=True, num_workers=4)
    model = MetricLearner(pretrain=args.pretrain)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    best_performace = np.Inf
    for epoch in range(0, args.epochs):
        model.train()

        loss = 0
        for i, batch in enumerate(dataset):
            batch = batch.to(device)
            embeddings = model(batch)

            optimizer.zero_grad()
            l = criterion.loss_func(embeddings)
            l.backward()
            optimizer.step()

            loss += l
            print('\tloss: %.4f'%(loss / (i+1)))
        loss /= (i+1)
        print('Batch %d\tloss:%.4f'%(epoch, loss))
        if loss < best_performace:
            torch.save({'state_dict': model.cpu().state_dict(), 'epoch': epoch, 'loss': loss}, \
                        os.path.join(args.ckpt, '%d_ckpt.pth'%epoch))
            shutil.copy(os.path.join(args.ckpt, '%d_ckpt.pth'%epoch), os.path.join(args.ckpt, 'best_performance.pth'))
            print('Saved model.')