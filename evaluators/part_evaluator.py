import torch
import numpy as np

class PartEvaluator():
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
				images = data['img']
				labels = data['label3d'].to(self.device)
				face_images = images[0].to(self.device)
				left_eye_images = images[1].to(self.device)
				right_eye_images = images[2].to(self.device)
				pred = model(face_images, left_eye_images, right_eye_images)
				loss = self.criterion(pred, labels)
				
				total_val_loss += loss.item() * 100

				distance = self.metric(pred, labels)
				avg_error += distance.item()

		angular_error = avg_error / len(self.val_dataset)
		val_loss = total_val_loss / len(self.val_dataset)

		return angular_error, val_loss