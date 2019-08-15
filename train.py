import argparse
import torch
import os
import sys
import shutil
import numpy as np
import visdom
import criterion
import cv2
import torchvision
import torchvision.transforms as transforms
import torchnet
from tensorboardX import SummaryWriter
from PIL import Image
from sampler import BalancedBatchSampler
from model import MetricLearner
from dataset import MetricData, SourceSampler, ImageFolderWithName, invTrans

eps = 1e-8
mlog = torchnet.logger.MeterLogger(env='logger')
writer = SummaryWriter()

def get_args():
    parser = argparse.ArgumentParser(description='Face Occlusion Regression')
    # train
    parser.add_argument('--pretrain', type=str, default='/root/.torch/models/googlenet-1378be20.pth', help='pretrain googLeNet model paht')
    parser.add_argument('--att-heads', type=int, default=8, help='number of attention modules')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs plans to train in total')
    parser.add_argument('--epoch_start', type=int, default=0, help='start epoch to count from')
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--batch_k', type=int, default=4, help='number of samples for a class of a batch')
    parser.add_argument('--num_batch', type=int, default=5000, help='number of batches per epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--ckpt', type=str, default='./ckpt', help='checkpoint folder')
    parser.add_argument('--resume', action='store_true', help='load previous best model and resume training')
    parser.add_argument('--num_workers', default=2, type=int, help='')
    # test
    parser.add_argument('--test', action='store_true', help='switch on test mode')
    # annotation
    parser.add_argument('--anno', type=str, required=True, help='location of annotation file')
    parser.add_argument('--anno_test', type=str, required=True, help='location of test data annotation file')
    parser.add_argument('--img_folder', type=str, required=True, help='folder of image files in annotation file')
    parser.add_argument('--img_folder_test', type=str, default='', help='folder of test image files in annotaion file')
    parser.add_argument('--idx_file', type=str, required=True, help='idx file for every label class')
    parser.add_argument('--idx_file_test', type=str, default='idx_file.pkl', help='idx file for test data, should be .pkl format')
    # model hyperparameter
    parser.add_argument('--in_size', type=int, default=128, help='input tensor shape to put into model')
    return parser.parse_args()

def imagefolder(folder, loader=lambda x: Image.open(x).convert('RGB'), return_fn=False):
    data = ImageFolderWithName(return_fn=return_fn, root=folder, transform=transforms.Compose([
                                        transforms.Resize(228),
                                        transforms.RandomCrop((224, 224)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])]),
                                        loader=loader)
    return data   

args = get_args()

device = torch.device('cuda:{}'.format(args.gpu_ids[0])) if args.gpu_ids else torch.device('cpu')
#data = MetricData(data_root=args.img_folder, anno_file=args.anno, idx_file=args.idx_file)
data = imagefolder(args.img_folder)
dataset = torch.utils.data.DataLoader(data, batch_sampler=BalancedBatchSampler(data, batch_size=args.batch, batch_k=args.batch_k, length=args.num_batch), num_workers=args.num_workers)
model = MetricLearner(pretrain=args.pretrain, batch_k=args.batch_k, att_heads=args.att_heads)
if args.resume:
    if args.ckpt.endswith('.pth'):
        state_dict = torch.load(args.ckpt)
    else:
        state_dict = torch.load(os.path.join(args.ckpt, 'best_performance.pth'))
    best_performace = state_dict['loss']
    start_epoch = state_dict['epoch'] + 1
    model.load_state_dict(state_dict['state_dict'], strict=False)
    print('Resume training. Start from epoch %d'%start_epoch)
else:
    start_epoch = 0
    best_performace = np.Inf
model = model.to(device)
att_params = [t[0] for t in model.att.named_parameters()]
optimizer = torch.optim.SGD([p for n, p in model.named_parameters() if n not in att_params], lr=args.lr, momentum=0.9)
optimizer_att = torch.optim.Adam(model.att.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-4)


if __name__ == '__main__':
    # TEST DATASET
    if args.test and args.resume:
        dataset_test = torch.utils.data.DataLoader(MetricData(args.img_folder_test, args.anno_test, args.idx_file_test, return_fn=True), \
                            batch_size=1, shuffle=True, drop_last=False, num_workers=max(1, int(args.num_workers/2)))
        vis = visdom.Visdom()
        model.eval()
        top_4 = {}
        with torch.no_grad():
            for i, batch in enumerate(dataset_test):
                batch[0] = batch[0].to(device)
                if i < 4:
                    query, atts = model(batch[0], ret_att=True)
                    imgs = MetricData.tensor2img(batch[0])
                    print('number of images in batch:', len(imgs), imgs[0].shape, imgs[0].min(), imgs[0].max())
                    tmp_att = atts[0].cpu().numpy().mean(axis=1)
                    print('number of attentions in batch:', len(atts), atts[0].shape, atts[0].min(), atts[0].max(), tmp_att.shape, tmp_att.min(), tmp_att.max())
                    for j in range(4):
                        vis.heatmap(cv2.resize(atts[j].cpu().numpy()[0, ...].mean(axis=0), (224, 224)), \
                            win=j+1000, opts=dict(title='Att_%d'%j))
                    '''
                    att_imgs = np.concatenate([np.transpose((np.repeat(atts[i].cpu().numpy()[0, ...].mean(axis=0)[...,np.newaxis], 3, axis=-1)*255).astype(np.uint8), (2, 0, 1))[np.newaxis] for i in range(3)])
                    print(att_imgs.shape)
                    vis.images(att_imgs, \
                        win=i+1000, opts=dict(title='Att_%d'%i))
                    '''
                    top_4[i] = {'fn': batch[1][0], 'query': query.cpu().numpy(), 'top_8': []}
                    vis.image(np.transpose(cv2.imread(os.path.join(args.img_folder_test, top_4[i]['fn']))[..., ::-1], (2, 0, 1)), \
                        win=i+100, opts=dict(title='Query_%d'%i))    
                    print('Added query.')                
                else:
                    embedding = model(batch[0]).cpu().numpy()
                    for j in range(4):
                        dist = np.sum((top_4[j]['query'] - embedding)**2)
                        if len(top_4[j]['top_8']) < 8 or (len(top_4[j]['top_8']) >= 8 and dist < top_4[j]['top_8'][-1]['distance']):
                            top_4[j]['top_8'].append({'fn': batch[1][0], 'distance': dist})
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

            loss_div, loss_homo, loss_heter = 0, 0, 0
            for i, batch in enumerate(dataset):
                x, y = batch
                x = x.to(device)
                out, atts = model(x, ret_att=True)
                a_indices, anchors, positives, negatives, _ = out
                # print(anchors.shape, positives.shape, negatives.shape, atts[0].shape)
                anchors, positives, negatives = torch.reshape(anchors, (-1, model.att_heads, int(512/model.att_heads))), torch.reshape(positives, (-1, model.att_heads, int(512/model.att_heads))), torch.reshape(negatives, (-1, model.att_heads, int(512/model.att_heads)))

                optimizer.zero_grad()
                optimizer_att.zero_grad()
                l_div, l_homo, l_heter = criterion.criterion(anchors, positives, negatives)
                l_div /= (model.att_heads - 1)
                l = l_div + 2*(l_homo + l_heter)
                l.backward()
                if i % 2 == 0:
                    optimizer_att.step()
                else:
                    optimizer_att.step()
                    optimizer.step()

                loss_homo += l_homo.item()
                loss_heter += l_heter.item()
                loss_div += l_div.item()
                if i % 100 == 0:
                    print('\tBatch %d\tloss div: %.4f (%.3f)\tloss homo: %.4f (%.3f)\tloss heter: %.4f (%.3f)'%\
                        (i, loss_div/(i+1), (loss_div+eps)/(loss_div+loss_heter+loss_homo+eps), loss_homo/(i+1), (loss_homo+eps)/(loss_div+loss_homo+loss_heter+eps), loss_heter/(i+1), (loss_heter+eps)/(loss_div+loss_heter+loss_homo+eps)))
                if i % 200 == 0:
                    img_inv = torch.cat([invTrans(x[i]).unsqueeze(0) for i in range(x.shape[0])], 0)
                    assert img_inv.shape == x.shape
                    writer.add_images('img', img_inv)
                    for ai in range(len(atts)):
                        writer.add_images('attention %d'%ai, atts[ai][:, 0:1, ...])
                    for var_name, value in model.att.named_parameters():
                        writer.add_histogram(var_name+'/grad', value.grad.data.cpu().numpy())
            loss_homo /= (i+1)
            loss_heter /= (i+1)
            loss_div /= (i+1)
            print('Epoch %d batches %d\tdiv:%.4f\thomo:%.4f\theter:%.4f'%(epoch, i+1, loss_div, loss_homo, loss_heter))
            # mlog.update_loss(loss_homo, 'homo')
            # mlog.update_loss(loss_heter, 'heter')
            # mlog.update_loss(loss_div, 'divergence')
            if (loss_homo+loss_heter) < best_performace:
                best_performace = loss_homo + loss_heter
                torch.save({'state_dict': model.cpu().state_dict(), 'epoch': epoch+1, 'loss': best_performace}, \
                            os.path.join(args.ckpt, '%d_ckpt.pth'%epoch))
                shutil.copy(os.path.join(args.ckpt, '%d_ckpt.pth'%epoch), os.path.join(args.ckpt, 'best_performance.pth'))
                print('Saved model.')
                model.to(device)
    except KeyboardInterrupt:
        torch.save({'state_dict': model.cpu().state_dict(), 'epoch': epoch+1, 'loss': best_performace}, \
                            os.path.join(args.ckpt, 'latest_ckpt.pth'))
        print('Save temporary model to latest_ckpt.pth')
        exit(0)
