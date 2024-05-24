import os
import torch
import numpy as np
import argparse
import random

def define_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, default = 'datasets/Gaze360/')
    parser.add_argument('--epochs', type = int, default = 200)
    parser.add_argument('--lr', type = float, default = 1e-4)
    parser.add_argument('--batch_size', type = int, default = 256)
    parser.add_argument('--num_workers', type = int, default = 6)
    parser.add_argument('--model', type = str, default = 'baseline')
    parser.add_argument('--pretrained', type = bool, default = False)
    parser.add_argument('--exp_index', type=int, default=100)
    parser.add_argument('--checkpoint', type=str, default = '')
    parser.add_argument('--loss', type=str, default = 'angular')
    parser.add_argument('--trainer', type=str, default = 'simple', choices=['simple', 'part'])
    parser.add_argument('--scheduler', type=str, default = 'step', choices=['step', 'one-cycle'])
    parser.add_argument('--run_name', type=str, default = '')
    parser.add_argument('--augmentation', type=str, default = 'none', choices=['none', 'randaugment', 'weak'])
    parser.add_argument('--magnitude', type=int, default = 5)
    parser.add_argument('--loss_alpha', type=float, default = 1)
    parser.add_argument('--seed', type=int, default = 42)
    parser.add_argument('--pretraining_dataset', type=str, choices=['imagenet', 'vggface2', 'casia-webface'], default='vggface2')
    parser.add_argument('--angle', type=int, choices=[90, 20], default=90) # For Front Facing vs Front 180
    args = parser.parse_args()

    return args

def gazeto3d(gaze):
    gaze_gt = np.zeros([3])
    gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
    gaze_gt[1] = -np.sin(gaze[1])
    gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
    return gaze_gt

def angular(gaze, label):
    total = np.sum(gaze * label)
    return np.arccos(min(total/(np.linalg.norm(gaze)* np.linalg.norm(label)), 0.9999999))*180/np.pi

def load_checkpoint(model, checkpoint_path):
	if os.path.isfile(checkpoint_path):
		print("Loading " + (checkpoint_path))
		model.load_state_dict(torch.load(checkpoint_path))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class AngularDistance(torch.nn.Module):
    def __init__(self):
        super(AngularDistance, self).__init__()
        self.sim = torch.nn.CosineSimilarity()

    def forward(self, gaze, label):
        return torch.sum(torch.acos(self.sim(gaze, label)) * 180 / torch.pi, dim=0)

class CosineSimilarityLoss(torch.nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, input1, input2):
        # Compute the cosine similarity
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        similarity = cos(input1, input2)

        # Compute the loss
        loss = 1 - similarity.mean()

        return loss

# Loss that combines MSE and Cosine Similarity
class MSEAngularLoss(torch.nn.Module):
    def __init__(self, alpha=0.5):
        super(MSEAngularLoss, self).__init__()
        self.alpha = alpha
        self.mse = torch.nn.MSELoss()
        self.cosine = CosineSimilarityLoss()

    def forward(self, input1, input2):
        mse_loss = self.mse(input1, input2)
        cosine_loss = self.cosine(input1, input2)
        total_loss = self.alpha * mse_loss + cosine_loss

        return total_loss

def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

# Reverse the image standardization
def reverse_image_standardization(image_tensor):
    processed_tensor = image_tensor * 128.0 + 127.5
    return processed_tensor
