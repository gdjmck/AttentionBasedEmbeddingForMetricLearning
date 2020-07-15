import torch
import torchvision
import random
from torch.utils.data.sampler import  Sampler
from scipy.special import comb
import numpy as np
import util

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, batch_k, length=None):
        assert (batch_size % batch_k == 0 ) and (batch_size > 0)
        self.dataset = {}
        if type(dataset) is torchvision.datasets.MNIST:
            self.dataset_type = 'mnist'
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            self.dataset_type = 'image_folder'
        elif dataset.name == 'in_shop':
            self.dataset_type = dataset.name
        else:
            self.dataset_type = 'not_supported'
        self.balanced_max = 0
        self.batch_size = batch_size
        self.batch_k = batch_k # number of classes in one batch
        self.length = length

        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = []
            self.dataset[label].append(idx)
        # eliminate all the data with less count than batch_k
        remove_keys = [key for key in self.dataset.keys() if len(self.dataset[key]) < self.batch_k]
        print('%d classes remove from dataset'%len(remove_keys))
        for remove_key in remove_keys:
            del self.dataset[remove_key]
        
        num_samples = [len(value) for value in self.dataset.values()]
        self.max_samples = max(num_samples)
        self.min_samples = min(num_samples)

        assert self.min_samples >= self.batch_k
    
        self.keys = list(self.dataset.keys())
        self.class_probs = torch.Tensor([1/len(self.keys)]*len(self.keys))
        self.class_indices = list(range(len(self.keys)))
        print('BalancedBatchSampler len:', self.__len__(), 'self.keys len=', len(self.keys))
        #self.currentkey = 0

    def __iter__(self):
        for i in range(self.__len__()):
            batch = []
            # classes = torch.multinomial(self.class_probs, int(self.batch_size/self.batch_k))
            classes = np.random.choice(self.class_indices, int(self.batch_size/self.batch_k), False)
            for cls in classes:
                cls_idxs = self.dataset[self.keys[cls]]
                # for k in torch.multinomial(torch.Tensor([1/len(cls_idxs)]*len(cls_idxs)), self.batch_k):
                batch += [cls_idxs[k] for k in np.random.choice(range(len(cls_idxs)), self.batch_k, False)]
                '''
                # 看是否list.append拖慢速度
                for k in np.random.choice(range(len(cls_idxs)), self.batch_k, False):
                    batch.append(cls_idxs[k])
                '''
            yield batch

    def __len__(self):
        if self.length is not None:
            return self.length
        return int(len(self.keys) * comb(self.min_samples, self.batch_k) / self.batch_size)
        
    def _get_label(self, dataset, idx):
        if self.dataset_type == 'mnist':
            return dataset.train_labels[idx].item()
        elif self.dataset_type == 'image_folder':
            return dataset.imgs[idx][1]
        elif self.dataset_type == 'in_shop':
            return dataset.get_attributes(dataset.list[idx])[-1]
        else:
            raise NotImplementedError
        
	
if __name__ == '__main__':
    import sys
    import torchvision.transforms as transforms
    '''
    dataset = torchvision.datasets.ImageFolder('/home/chengk/chk/car_kaggle/train', 
                                        transform=transforms.Compose([
                                            transforms.Resize(228),
                                            transforms.RandomCrop((224, 224)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])]))
    '''
    import dataset
    dataset = dataset.InShop('/home/chengk/chk/DeepFashion/In-shop/DATA_v2/In-shop Clothes Retrieval Benchmark/Img/',
                            '/home/chengk/chk/DeepFashion/In-shop/In-shop Clothes Retrieval Benchmark/Eval/list_eval_partition.txt')
    batch, batch_k = 4, 2
    num_batch = 10
    sampler = BalancedBatchSampler(dataset, batch, batch_k, num_batch)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    print(len(loader))
    print('start sampling from BalancedBatchSampler:')
    for i in range(2):
        for batch_ in sampler.__iter__():
            for group in range(batch//batch_k):
                if len(np.unique(batch_[group*batch_k: (group+1)*batch_k])) != batch_k:
                    print('duplicate sample')
                print(dataset.get_attributes(dataset.list[batch_[group*batch_k]])[-1],
                    dataset.get_attributes(dataset.list[batch_[group*batch_k+1]])[-1])
    sys.exit(0)