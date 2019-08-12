import torchvision
from PIL import Image
import train
import os

def loader_test(fn):
    parts = fn.split('/')
    label = parts[-2]
    return fn, label, Image.open(fn).convert('RGB')

if __name__ == '__main__':
    args = train.args
    model = train.model

    data = train.convert_dataset(os.path.join(args.img_folder_test, 'train'), loader_test)
    print('class to idx:', data.class_to_idx)
    print(len(data.targets), data.targets)
