import torchvision
from PIL import Image
import train

def loader_test(fn):
    parts = fn.split('/')
    label = parts[-2]
    return fn, label, Image.open(fn).convert('RGB')

if __name__ == '__main__':
    args = train.args
    model = train.model

    data = train.convert_dataset(args.img_folder_test, loader_test)
    for item in iter(data):
        print(item)
        break