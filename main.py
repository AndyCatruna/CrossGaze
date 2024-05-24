import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from models import *
from utils import *
from helpers import *
from evaluators import *
import numpy as np
import random

# Arguments
args = define_args()
config = vars(args)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Hyperparameters
workers = args.num_workers
epochs = args.epochs
batch_size = args.batch_size
lr = args.lr

run_name = args.run_name

def main():
	# For logging and saving checkpoints
	os.makedirs('weights', exist_ok=True)
	os.makedirs('results', exist_ok=True)
	f = open('results/' + run_name + '.txt', 'w')

	# Augmentation
	train_transforms, test_transforms = get_transforms(args)

	# Loaders
	train_dataset, train_loader, val_dataset, val_loader = get_loaders(args, train_transforms, test_transforms)

	# Model
	model = choose_model(args)

	print(args.model + ": Number of Parameters - " + str(count_parameters(model)))
	if args.checkpoint:
		load_checkpoint(model, args.checkpoint)

	model= nn.DataParallel(model)
	model.to(device)

	# Evaluation Metric and Loss
	metric = AngularDistance()

	if args.loss == 'mse':
		criterion = nn.MSELoss().cuda()
	elif args.loss == 'angular':
		criterion = CosineSimilarityLoss().cuda()
	elif args.loss == 'mse-angular':
		criterion = MSEAngularLoss(args.loss_alpha).cuda()

	# Optimizer and Scheduler
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
	scheduler = get_scheduler(args, optimizer, train_loader)

	# Trainer and Evaluator
	trainer = get_trainer(args, train_loader, device, optimizer, scheduler, criterion)

	if args.trainer == 'part':
		evaluator = PartEvaluator(args, val_dataset, val_loader, device, criterion, metric)
	else:
		evaluator = SimpleEvaluator(args, val_dataset, val_loader, device, criterion, metric)

	# Training Loop
	min_angular_error = 100
	print(run_name)
	for epoch in range(epochs):
		print("EPOCH " + str(epoch))
		train_loss = trainer.train(model)
		
		angular_error, val_loss = evaluator.eval(model)
		print("VAL LOSS: " + str(val_loss))
		print("ANGULAR ERROR: " + str(angular_error) + " Current Best: " + str(min_angular_error) + "\n")

		f = open('results/' + run_name + '.txt', 'a')
		f.write(str(angular_error) + '\n')
		f.close()

		if angular_error < min_angular_error:
			min_angular_error = angular_error
			print("Saving")
			checkpoint_path = 'weights/' + run_name + '.pth'
			torch.save(model.state_dict(), checkpoint_path)

	print("-" * 50)
	print("Finished Training")
	print("Best Angular Error: " + str(min_angular_error))
	print("-" * 50, "\n")

if __name__ == '__main__':
	main()