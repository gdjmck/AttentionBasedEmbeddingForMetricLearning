import pickle
import scipy.io as sio
import torch
import os
from PIL import Image
import torchvision.transforms as transforms

class MetricData(torch.utils.data.Dataset):
    def __init__(self, data_root, anno_file, idx_file):
        if idx_file.endswith('pkl'):
            with open(idx_file, 'rb') as f:
                self.idx = pickle.laod(f)
        assert anno_file.endswith('mat')
        self.anno = sio.loadmat(anno_file)['annotations']
        self._convert_labels()
        self.data_root = data_root
        self.transforms = transforms.Compose([transforms.Resize(256), transforms.RandomCrop((224, 224)), \
                                                transforms.RandomHorizontalFlip(), transforms.ToTensor(), \
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return self.anno.shape[1]
    
    def _convert_labels(self):
        labels, fns = [], []
        for i in self.anno.shape[1]:
            labels.append(self.anno[0, i][-2][0, 0])
            fns.append(self.anno[0, i][-1][0])
        self.labels = labels
        self.fns = fns

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.data_root, self.fns[i])).convert('RGB')
        img = F.resize(img, 256)
        img = self.transforms(img)
        return img

class SourceSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size=32):
        self.data_source = data_source
        self.num_samples = len(self.data_source)
        self.batch_size = batch_size
        self.idx_dict = self.data_source.idx

    def __len__(self):
        return self.num_samples * self.batch_size * 2

    def __iter__(self):
        indices = torch.randperm(self.num_samples)
        ret_idx = []
        for i, idx in enumerate(indices):
            label = self.data_source.labels[idx]
            if i % self.batch_size < self.batch_size / 2: # positive pair
                idx_list = self.idx_dict[label]
                s = np.random.choice(idx_list, size=2, replace=False)
                ret_idx.extend([idx, s[1]] if s[0] == idx else s)
            else: # negative pair
                # 从非idx中抽一个label
                neg_labels = np.random.choice(self.data_source.idx_dict.keys(), 2, replace=False)
                neg_label = neg_labels[0] if neg_labels[0] != label else neg_labels[1]
                s = np.random.choice(self.idx_dict[neg_label], 1)
                ret_idx.extend([idx, s[0]])
        return iter(ret_idx)


if __name__ == '__main__':
    dataset = torch.utils.data.DataLoader(MetricData('/home/chk/cars_stanford/devkit/cars_train_annos.mat'))