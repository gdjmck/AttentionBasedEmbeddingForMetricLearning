import pickle
import scipy.io as sio
import torch
import os
import numpy as np
from PIL import Image
from scipy.special import comb
import torchvision.transforms as transforms

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

class MetricData(torch.utils.data.Dataset):
    def __init__(self, data_root, anno_file, idx_file, return_fn=False):
        self.return_fn = return_fn
        if idx_file.endswith('pkl'):
            with open(idx_file, 'rb') as f:
                self.idx = pickle.load(f)
        assert anno_file.endswith('mat')
        self.anno = sio.loadmat(anno_file)['annotations']
        self._convert_labels()
        self.data_root = data_root
        self.transforms = transforms.Compose([transforms.Resize(256), transforms.RandomCrop((224, 224)), \
                                                transforms.RandomHorizontalFlip(), transforms.ToTensor(), \
                                                transforms.Normalize(mean=mean,std=std)])

    def __len__(self):
        return self.anno.shape[1]
    
    def _convert_labels(self):
        labels, fns = [], []
        for i in range(self.anno.shape[1]):
            labels.append(self.anno[0, i][-2][0, 0])
            fns.append(self.anno[0, i][-1][0])
        self.labels = labels
        self.fns = fns

    @classmethod
    def tensor2img(cls, tensor):
        if type(tensor) != np.ndarray:
            tensor = tensor.cpu().numpy()

        if len(tensor.shape) == 4:
            imgs = []
            for i in range(tensor.shape[0]):
                imgs.extend(cls.tensor2img(tensor[i, ...]))
            return imgs
        assert tensor.shape[0] == 3
        img = np.transpose(tensor, (1, 2, 0))
        img = img * np.array(std) + np.array(mean)
        return [img*255]


    def __getitem__(self, i):
        # print('__getitem__\t', i%16, '\tlabel:', self.labels[i])
        #label = self.labels[i]
        img = Image.open(os.path.join(self.data_root, self.fns[i])).convert('RGB')
        img = self.transforms(img)
        return img if not self.return_fn else (img, self.fns[i])

class SourceSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_k=2, batch_size=32):
        self.data_source = data_source
        self.batch_k = batch_k
        self.num_samples = len(self.data_source)
        self.batch_size = batch_size
        self.idx_dict = self.data_source.idx

        
        labels, num_samples = np.unique(self.data_source.labels, return_counts=True)
        self.max_samples = max(num_samples)
        self.min_samples = min(num_samples)
        self.labels = labels

        assert self.min_samples >= self.batch_k

    def __len__(self):
        # return self.num_samples * self.batch_size * 2
        iter_len = len(self.labels) * (comb(self.max_samples, self.batch_k) + comb(self.min_samples, self.batch_k))
        return iter_len - iter_len % self.batch_size

    def __iter__(self):
        while(True):
            # sample both positive and negative labels
            pos_labels = np.random.choice(self.labels, int(self.batch_size/(2*self.batch_k)), replace=False)
            neg_labels = np.random.choice(self.labels, int(self.batch_size/(2*self.batch_k)), replace=False)
            ret_idx = []
            for label in pos_labels:
                idx_list = self.idx_dict[label]
                ret_idx.extend(np.random.choice(idx_list, 2, replace=False))
            for label in neg_labels:
                ret_idx.extend(np.random.choice(self.idx_dict[label], 2, replace=False))
            yield ret_idx


if __name__ == '__main__':
    data = MetricData(data_root='/home/chk/cars_stanford/cars_train', \
                                    anno_file='/home/chk/cars_stanford/devkit/cars_train_annos.mat', \
                                    idx_file='/home/chk/cars_stanford/devkit/cars_train_annos_idx.pkl', \
                                    return_fn=True)
    dataset = torch.utils.data.DataLoader(data, batch_sampler=SourceSampler(data))

    from model import MetricLearner
    model = MetricLearner()
    for label, td in dataset:
        print('Batch shape:\t', td.shape, '\t', label)
        pred = model(td)
        print(pred.shape)
        break