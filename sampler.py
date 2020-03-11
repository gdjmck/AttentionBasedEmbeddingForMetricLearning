import torch
import torchvision
import random
from torch.utils.data.sampler import  Sampler
from scipy.special import comb
import numpy as np

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, batch_k, length=None):
        assert (batch_size % batch_k == 0 ) and (batch_size > 0)
        self.dataset = {}
        self.balanced_max = 0
        self.batch_size = batch_size
        self.batch_k = batch_k
        self.length = length

        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = []
            self.dataset[label].append(idx)
        
        num_samples = [len(value) for value in self.dataset.values()]
        self.max_samples = max(num_samples)
        self.min_samples = min(num_samples)

        assert self.min_samples >= self.batch_k
    
        self.keys = list(self.dataset.keys())
        self.class_probs = torch.Tensor([1/len(self.keys)]*len(self.keys))
        print('BalancedBatchSampler len:', self.__len__(), 'self.keys len=', len(self.keys))
        #self.currentkey = 0

    def __iter__(self):
        for i in range(self.__len__()):
            batch = []
            classes = torch.multinomial(self.class_probs, int(self.batch_size/self.batch_k))
            for cls in classes:
                cls_idxs = self.dataset[self.keys[cls]]
                for k in torch.multinomial(torch.Tensor([1/len(cls_idxs)]*len(cls_idxs)), self.batch_k):
                    batch.append(cls_idxs[k])
            yield batch

    def __len__(self):
        if self.length is not None:
            return self.length
        return int(len(self.keys) * comb(self.min_samples, self.batch_k) / self.batch_size)
        
    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        else:
            raise NotImplementedError
        
	
if __name__ == '__main__':
    import sys
    import torchvision.transforms as transforms
    dataset = torchvision.datasets.ImageFolder('/home/chk/Downloads/cars_stanford/car_kaggle/train', 
                                        transform=transforms.Compose([
                                            transforms.Resize(228),
                                            transforms.RandomCrop((224, 224)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])]))
    sampler = BalancedBatchSampler(dataset, 4, 2, 100)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    print(len(loader))
    sys.exit(0)
    for i, _ in enumerate(loader):
        pass
    print(i+1)