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
    att_heads = args.att_heads
    batchsize = args.batch
    model = train.model
    model.eval()

    data = train.imagefolder(args.img_folder_test)
    # print('class to idx:', data.class_to_idx)
    # print(len(data.targets), data.targets)

    dataset = torch.utils.data.DataLoader(data, batch_size=batchsize)
    #dataset = [torch.load('../AttentionEmbedding/ckpt_4head/best_performance.pth')['data']]
    embeddings = {}
    with torch.no_grad():
        for (img, label) in dataset:
            img = img.to(train.device)
            embedding = model(img, sampling=False)
            embedding = embedding.view(batchsize, att_heads, -1)
            #embedding = train.F.normalize(embedding, 2, -1)
            embedding = embedding.view(-1, 512).cpu().numpy()
            for i, l in enumerate(label):
                l = l.item()
                #print(embedding.shape)
                if l not in embeddings.keys():
                    embeddings[l] = [embedding[i: i+1, ...]]
                else:
                    embeddings[l].append(embedding[i: i+1, ...])
        
    with open(args.ckpt.rsplit('/', 1)[0] + '/embeddings_cars196_epoch%s.pkl'%args.ckpt.rsplit('/', 1)[-1].split('_')[0], 'wb') as f:
        pickle.dump(embeddings, f)
        print('saved embedding.')
