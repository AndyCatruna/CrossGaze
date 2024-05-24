import torch
import numpy as np

class SimpleTrainer():
    def __init__(self, args, train_loader, device, optimizer, scheduler, criterion):
        self.args = args
        self.train_loader = train_loader
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

    def train(self, model):
        model.train()
        total_train_loss = 0
        for i, data in enumerate(self.train_loader):
            images = data['img'].to(self.device)
            labels = data['label3d'].to(self.device)
            
            self.optimizer.zero_grad()
            pred = model(images)
            loss = self.criterion(pred, labels)
            
            loss.backward()
            self.optimizer.step()

            total_train_loss += loss.item() * 100

            if self.args.scheduler == 'one-cycle':
                self.scheduler.step()

        if self.args.scheduler != 'one-cycle':
            self.scheduler.step()

        train_loss = np.round(total_train_loss / len(self.train_loader), 2)
        print("TRAIN LOSS: " + str(train_loss))
        
        return train_loss
