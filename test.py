import torch
import torchvision
import pickle
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

    data = train.convert_dataset(os.path.join(args.img_folder_test, 'train'))
    # print('class to idx:', data.class_to_idx)
    # print(len(data.targets), data.targets)

    dataset = torch.utils.data.DataLoader(data)
    embeddings = {}
    for i, (img, label) in enumerate(dataset):
        label = label.numpy()[0]
        print(label)
        assert label == data.targets[i]
        img = img.to(train.device)
        embedding = model(img, sampling=False).cpu().numpy()
        if label not in embeddings.keys():
            embeddings[label] = [embedding]
        else:
            embeddings[label].append(embedding)
    
    with open('embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
        print('saved embedding.')        