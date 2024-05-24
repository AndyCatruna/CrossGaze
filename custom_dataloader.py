'''
    Code based on: https://github.com/Ahmednull/L2CS-Net/blob/main/l2cs/datasets.py
'''

import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, path, root, transform, angle, args, train=True):
        self.transform = transform
        self.root = root
        self.orig_list_len = 0
        self.angle = angle
        self.args = args
        self.train = train

        self.lines = []
        if isinstance(path, list):
            for i in path:
                with open(i) as f:
                    print("here")
                    line = f.readlines()
                    line.pop(0)
                    self.lines.extend(line)
        else:
            with open(path) as f:
                lines = f.readlines()
                lines.pop(0)
                self.orig_list_len = len(lines)
                for line in lines:
                    gaze2d = line.strip().split(" ")[5]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs((label[0]*180/np.pi)) <= angle and abs((label[1]*180/np.pi)) <= angle:
                        self.lines.append(line)         
        
        print("{} items removed from dataset that have an angle > {}".format(self.orig_list_len-len(self.lines), angle))


    def __len__(self):
        return len(self.lines)

    def __get_data__(self, line):
        line = line.strip().split(" ")
        face = line[0]

        left = line[1]
        left = left.replace("Left", "NewLeft")
        
        right = line[2]
        right = right.replace("Right", "NewRight")
        
        name = line[3]
        
        gaze2d = line[5]
        label2d = np.array(gaze2d.split(",")).astype("float")
        label2d = torch.from_numpy(label2d).type(torch.FloatTensor)

        gaze3d = line[4]
        label3d = np.array(gaze3d.split(",")).astype("float")
        label3d = torch.from_numpy(label3d).type(torch.FloatTensor)

        identity = int(line[6])
        identity = torch.from_numpy(np.array(identity)).type(torch.LongTensor)

        left_eye = line[7]
        left_eye = np.array(left_eye.split(",")).astype("float")
        
        right_eye = line[8]
        right_eye = np.array(right_eye.split(",")).astype("float")

        return face, left, right, label2d, label3d, name, identity, left_eye, right_eye

    def __getinfo__(self, line):
        
        face, left, right, label2d, label3d, name, identity, left_eye, right_eye = self.__get_data__(line)
        img = Image.open(os.path.join(self.root, face))

        if self.args.trainer == 'part':
            left_eye_image = Image.open(os.path.join(self.root, left))
            right_eye_image = Image.open(os.path.join(self.root, right))

            transform, eye_transform = self.transform
            if self.transform:
                img = transform(img)
                left_eye_image = eye_transform(left_eye_image)
                right_eye_image = eye_transform(right_eye_image)
                img = [img, left_eye_image, right_eye_image]

        elif self.transform:
            img = self.transform(img)
        
        return img, label2d, label3d, name, identity, left_eye, right_eye

    def __getitem__(self, idx):
        line = self.lines[idx]
        img, label2d, label3d, name, identity, left_eye, right_eye = self.__getinfo__(line)
       
        output = {'img': img, 'label2d': label2d, 'label3d': label3d, 'name': name, 'identity': identity, 'left_eye': left_eye, 'right_eye': right_eye}

        return output
