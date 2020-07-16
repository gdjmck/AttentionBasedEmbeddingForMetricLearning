import argparse
import torch
import os
import sys
import math
import shutil
import numpy as np
import visdom
import criterion
import cv2
import util
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from OneCycle import get_lr
import torch.nn.functional as F
import time
from tensorboardX import SummaryWriter
from PIL import Image
from sampler import BalancedBatchSampler
from model import MetricLearner
from dataset import MetricData, SourceSampler, ImageFolderWithName, invTrans
from torchsummary import summary

eps = 1e-8
lambda_reg = 0.02
sampling = False

def get_args():
    parser = argparse.ArgumentParser(description='Face Occlusion Regression')
    # train
    parser.add_argument('--pretrain', type=str, default='/root/.torch/models/googlenet-1378be20.pth', help='pretrain googLeNet model path')
    parser.add_argument('--att-heads', type=int, default=8, help='number of attention modules')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use empty string for CPU')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs plans to train in total')
    parser.add_argument('--epoch_start', type=int, default=0, help='start epoch to count from')
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--batch_k', type=int, default=4, help='number of samples for a class of a batch')
    parser.add_argument('--num_batch', type=int, default=5000, help='number of batches per epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--ckpt', type=str, default='./ckpt', help='checkpoint folder')
    parser.add_argument('--tb', type=str, default='./runs', help='tensorboard folder')
    parser.add_argument('--resume', action='store_true', help='load previous best model and resume training')
    parser.add_argument('--ignore', action='store_true', help='ignore the best loss in checkpoint')
    parser.add_argument('--cycle', action='store_true', help='turn on one cycle policy')
    parser.add_argument('--num_workers', default=2, type=int, help='')
    # test
    parser.add_argument('--test', action='store_true', help='switch on test mode')
    parser.add_argument('--find-lr', action='store_true', help='find a suitable lr for training.')
    # annotation
    parser.add_argument('--img_folder', type=str, required=True, help='folder of image files in annotation file')
    parser.add_argument('--img_folder_test', type=str, default='', help='folder of test image files in annotaion file')
    # model hyperparameter
    parser.add_argument('--in_size', type=int, default=128, help='input tensor shape to put into model')
    return parser.parse_args()

def imagefolder(folder, loader=lambda x: Image.open(x).convert('RGB'), return_fn=False):
    data = ImageFolderWithName(return_fn=return_fn, root=folder, transform=transforms.Compose([
                                        transforms.ColorJitter(0.2, 0.1, 0.1, 0.1),
                                        transforms.Resize(235),
                                        transforms.RandomCrop((224, 224)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])]),
                                        loader=loader)
    return data   

args = get_args()
if not os.path.exists(args.tb):
    os.makedirs(args.tb)
writer = SummaryWriter(args.tb)

if args.gpu_ids:
    device = torch.device('cuda:{}'.format(args.gpu_ids[0]))
    cudnn.benchmark = True
    print('GPU accelerated.')
else:
    device = torch.device('cpu')
    print('Using CPU only.')

#dataset_test = torch.utils.data.DataLoader(data_test, batch_sampler=BalancedBatchSampler(data_test, batch_size=args.batch, batch_k=args.batch_k, length=args.num_batch//2))
use_att = args.att_heads > 1
model = MetricLearner(pretrain=args.pretrain, normalize=True, batch_k=args.batch_k, att_heads=args.att_heads, use_att=use_att)
reg_params = ['inception4a', 'inception4b', 'inception4c', 'inception4d', 'inception4e']
if not os.path.exists(args.ckpt):
    os.makedirs(args.ckpt)
    print('Init ', args.ckpt)
if args.resume:
    if args.ckpt.endswith('.pth'):
        state_dict = torch.load(args.ckpt)
    else:
        state_dict = torch.load(os.path.join(args.ckpt, 'best_performance.pth'))
    if args.ignore:
        best_performance = np.Inf
    else:
        best_performance = state_dict['loss']
    start_epoch = state_dict['epoch'] + 1
    model.load_state_dict(state_dict['state_dict'])#, strict=False
    print('Resume training. Start from epoch %d'%start_epoch)
else:
    start_epoch = 0
    best_performance = np.Inf
model = model.to(device)
#summary(model, (3, 224, 224))
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.95, weight_decay=1e-4)



step = 0
if __name__ == '__main__':
    #data = MetricData(data_root=args.img_folder, anno_file=args.anno, idx_file=args.idx_file)
    data = imagefolder(args.img_folder)
    #data_test = imagefolder(args.img_folder_test)
    dataset = torch.utils.data.DataLoader(data, batch_sampler=BalancedBatchSampler(data, batch_size=args.batch, batch_k=args.batch_k, length=args.num_batch), \
                                            num_workers=args.num_workers, pin_memory=True)

    # TEST DATASET
    if args.test and args.resume:
        dataset_test = torch.utils.data.DataLoader(imagefolder(return_fn=True, folder=args.img_folder_test),\
                                                    batch_size=1, shuffle=True, drop_last=False, num_workers=max(1, int(args.num_workers/2)))
        vis = visdom.Visdom()
        model.eval()
        top_4 = {}
        with torch.no_grad():
            for i, batch in enumerate(dataset_test):
                batch[0] = batch[0].to(device)
                if i < 4:
                    query, atts = model(batch[0], ret_att=True, sampling=False)
                    imgs = MetricData.tensor2img(batch[0])
                    print('number of images in batch:', len(imgs), imgs[0].shape, imgs[0].min(), imgs[0].max())
                    tmp_att = atts[:, 0, ...].cpu().numpy().mean(axis=1)
                    print('number of attentions in batch:', atts.shape[1], atts[:, 0, ...].shape, atts[:, 0, ...].min(), atts[:, 0, ...].max(), tmp_att.shape, tmp_att.min(), tmp_att.max())
                    for j in range(args.att_heads):
                        vis.heatmap(cv2.resize(atts[:, j, ...].cpu().numpy()[0, ...].mean(axis=0), (224, 224)), \
                            win=j+1000, opts=dict(title='Att_%d'%j))
                    '''
                    att_imgs = np.concatenate([np.transpose((np.repeat(atts[:, i, ...].cpu().numpy()[0, ...].mean(axis=0)[...,np.newaxis], 3, axis=-1)*255).astype(np.uint8), (2, 0, 1))[np.newaxis] for i in range(3)])
                    print(att_imgs.shape)
                    vis.images(att_imgs, \
                        win=i+1000, opts=dict(title='Att_%d'%i))
                    '''
                    top_4[i] = {'fn': batch[2][0][0], 'query': query.cpu().numpy(), 'top_8': []}
                    vis.image(np.transpose(cv2.imread(os.path.join(args.img_folder_test, top_4[i]['fn']))[..., ::-1], (2, 0, 1)), \
                        win=i+100, opts=dict(title='Query_%d'%i))    
                    print('Added query.')                
                else:
                    embedding = model(batch[0], sampling=False).cpu().numpy()
                    for j in range(4):
                        dist = np.sum((top_4[j]['query'] - embedding)**2)
                        if len(top_4[j]['top_8']) < 8 or (len(top_4[j]['top_8']) >= 8 and dist < top_4[j]['top_8'][-1]['distance']):
                            top_4[j]['top_8'].append({'fn': batch[2][0][0], 'distance': dist})
                            if len(top_4[j]['top_8']) > 8:
                                last_fn = top_4[j]['top_8'][-1]['fn']
                                top_4[j]['top_8'] = sorted(top_4[j]['top_8'], key=lambda x: x['distance'])
                                print('%d Sorted.'%j, top_4[j]['top_8'])
                                top_4[j]['top_8'] = top_4[j]['top_8'][:8]
                                update = False
                                for d in top_4[j]['top_8']:
                                    if d['fn'] == last_fn:
                                        update = True
                                        print('\nUpdated\n')
                                        break
                                if update:
                                    imgs = np.concatenate([np.transpose(cv2.resize(cv2.imread(os.path.join(args.img_folder_test, d['fn'])), (250, 250))[..., ::-1], (2, 0, 1))[np.newaxis] for d in top_4[j]['top_8']])
                                    vis.images(imgs, win=j, nrow=2, opts=dict(title='IMG_%d'%j))

        for item in top_4.values():
            print(item['fn'], '\n', item['top_8'], '\n\n')
        sys.exit()

    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()

            loss_div, loss_homo, loss_heter, loss_center, loss_reg = 0, 0, 0, 0, 0
            ticktime = time.time()
            #batch = next(iter(dataset))
            for i, batch in enumerate(dataset):
            #for i in range(len(dataset)):
                load_time = time.time() - ticktime
                x, y = batch
                x = x.to(device)

                forward_time = time.time()
                ret = model(x, sampling=sampling, ret_att=True)
                if use_att:
                    embeddings, atts = ret
                    atts_regularizer = 100*criterion.exclusion_loss(atts)
                    print('criterion_loss: ', atts_regularizer.item())
                else:
                    embeddings = ret
                forward_time = time.time() - forward_time
                if sampling:
                    _, anchors, positives, negatives, _ = embeddings
                    anchors = F.normalize(anchors.view(anchors.size(0), args.att_heads, -1), 2, -1)
                    positives = F.normalize(positives.view(positives.size(0), args.att_heads, -1), 2, -1)
                    negatives = F.normalize(negatives.view(negatives.size(0), args.att_heads, -1), 2, -1)
                else:
                    # switch off normalize and add it to a weight term in constraining vector size to be 1
                    # embeddings = F.normalize(embeddings.view(embeddings.size(0), args.att_heads, -1), 2, -1)
                    embeddings = embeddings.view(embeddings.size(0), args.att_heads, -1)

                #l_div, l_homo, l_heter = criterion.criterion(anchors, positives, negatives)
                loss_time = time.time()
                if sampling:
                    l_div, l_homo, l_heter = criterion.criterion(anchors, positives, negatives)
                else:
                    l_div, l_homo, l_heter = criterion.loss_func(embeddings, args.batch_k)
                l_metric = l_homo + l_heter
                if use_att:
                    l_metric += l_div
                loss_time = time.time() - loss_time
                #l_reg = lambda_reg * criterion.regularization(model, reg_params)
                #l_reg = torch.Tensor([0])
                optimizer.zero_grad()
                l_metric.backward(retain_graph=True)

                optimizer.step()

                #print('metric att:', torch.Tensor([var.grad.abs().mean() for var in model.att.parameters() if var.requires_grad]).mean())

                loss_homo += l_homo.item()
                loss_heter += l_heter.item()
                loss_div += l_div.item()
                #loss_reg += l_reg.item()
                #loss_center += l_centers.item()
                total_time = time.time() - ticktime
                writer.add_scalars(main_tag='TrainLog', tag_scalar_dict={'homo': l_homo.item(), 'heter': l_heter.item(), 'div': l_div.item()},
                                    global_step=step)
                if use_att and (1+i) % 50 == 0:
                    writer.add_scalar('att_mean', atts.mean().item(), global_step=step)
                if (1+i) % 50 == 0:
                    print('LR:', get_lr(optimizer))
                    print('\tBatch %d\tloss div: %.4f (%.3f)\tloss homo: %.4f (%.3f)\tloss heter: %.4f (%.3f)'%\
                        (i, l_div.item(), loss_div/(i+1), l_homo.item(), loss_homo/(i+1), l_heter.item(), loss_heter/(i+1)))
                # 各层的梯度
                if (i+1) % 100 == 0:
                    writer.add_figure('grad_flow', util.plot_grad_flow_v2(model.named_parameters()), global_step=step//5)
                if use_att and (i+1) % 200 == 0:
                    img_inv = torch.cat([invTrans(x[i]).unsqueeze(0) for i in range(x.shape[0])], 0)
                    assert img_inv.shape == x.shape
                    writer.add_images('img', img_inv, global_step=step)
                    for ai in range(model.att_heads):
                        writer.add_images('avg_attention %d'%ai, atts[:, ai, ...].mean(dim=1, keepdim=True))
                        writer.add_images('attention %d'%ai, atts[:, ai, 0:1, ...], global_step=step)
                    step += 1
                ticktime = time.time()
            loss_homo /= (i+1)
            loss_heter /= (i+1)
            loss_div /= (i+1)
            print('Epoch %d batches %d\tdiv:%.4f\thomo:%.4f\theter:%.4f'%(epoch, i+1, loss_div, loss_homo, loss_heter))
            writer.add_scalars(main_tag='Train', tag_scalar_dict={'homo': loss_homo, 'heter': loss_heter, 'div': loss_div},
                                global_step=epoch)

            # TEST PHASE
            '''
            model.eval()
            loss_div, loss_homo, loss_heter = 0, 0, 0
            for i, batch in enumerate(dataset_test):
                x, y = batch
                x = x.to(device)
                with torch.no_grad():
                    embeddings, atts = model(x, ret_att=True, sampling=False)
                embeddings = embeddings.view(embeddings.size(0), args.att_heads, -1)

                l_div, l_homo, l_heter = criterion.loss_func(embeddings, args.batch_k)
                loss_homo += l_homo.item()
                loss_heter += l_heter.item()
                loss_div += l_div.item()
            print('\tTest phase %d samples\tloss div: %.3f\tloss homo: %.3f\tloss heter: %.3f'%\
                (i, loss_div/(i+1), loss_homo/(i+1), loss_heter/(i+1)))
            writer.add_scalars(main_tag='Val', tag_scalar_dict={'homo': loss_homo/(i+1), 'heter': loss_heter/(i+1), 'div': loss_div/(i+1)},
                                global_step=epoch)

            '''
            if (loss_homo+loss_heter+loss_div) < best_performance:
                best_performance = loss_homo + loss_heter + loss_div
                dst_path = args.ckpt.rsplit('/', 1)[0] if args.ckpt.endswith('.pth') else args.ckpt
                torch.save({'state_dict': model.cpu().state_dict(), 'epoch': epoch+1, 'loss': best_performance,
                            'data': batch}, \
                            os.path.join(dst_path, '%d_ckpt.pth'%epoch))
                shutil.copy(os.path.join(dst_path, '%d_ckpt.pth'%epoch), os.path.join(dst_path, 'best_performance.pth'))
                print('Saved model.')
                model.to(device)

    except KeyboardInterrupt:
        if os.path.isfile(args.ckpt):
            temp_ckpt = os.path.join(args.ckpt.rsplit('/', 1)[0], 'latest_ckpt.pth')
        else:
            temp_ckpt = os.path.join(args.ckpt, 'latest_ckpt.pth')
        torch.save({'state_dict': model.cpu().state_dict(), 'epoch': epoch+1, 'loss': best_performance}, \
                            temp_ckpt)
        print('Save temporary model to latest_ckpt.pth')
        exit(0)
