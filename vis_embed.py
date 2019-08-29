import torch
from tensorboardX import SummaryWriter
import pickle
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Visualize embedding in tensorboard.')
	parser.add_argument('--embeddings', type=str, required=True, help='embedding pickle file.')
	parser.add_argument('--save-folder', type=str, default='./runs', help='directory to save tensorboard file.')
	return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()
	assert args.embeddings.endswith('.pkl')
	writer = SummaryWriter(args.save_folder)
	with open(args.embeddings, 'rb') as f:
		data = pickle.load(f)
	labels = []
	feats = []
	for label in data.keys():
		labels.extend([label]*len(data[label]))
		feats.extend(data[label])

	feats = [torch.Tensor(feat) for feat in feats]
	feats = torch.cat(feats, 0)
	print(feats.shape, len(labels))
	writer.add_embedding(feats, metadata=labels)
	writer.close()
