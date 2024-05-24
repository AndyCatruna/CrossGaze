import torch
import numpy as np

class SimpleEvaluator():
	def __init__(self, args, val_dataset, val_loader, device, criterion, metric):
		self.args = args
		self.val_loader = val_loader
		self.device = device
		self.criterion = criterion
		self.metric = metric
		self.val_dataset = val_dataset
	
	def eval(self, model):
		model.eval()
		total_val_loss = 0
		avg_error = 0
		with torch.no_grad():
			for i, data in enumerate(self.val_loader):
				images = data['img'].to(self.device)
				labels = data['label3d'].to(self.device)
				
				if self.args.model == 'debias':
					_, pred = model(images)
				else:
					pred = model(images)
				loss = self.criterion(pred, labels)
				
				total_val_loss += loss.item() * 100

				distance = self.metric(pred, labels)
				avg_error += distance.item()

		angular_error = avg_error / len(self.val_dataset)
		val_loss = total_val_loss / len(self.val_dataset)

		return angular_error, val_loss