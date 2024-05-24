from models import *
from trainers import *
import torchvision.transforms as transforms
from utils import fixed_image_standardization
from custom_dataloader import *
import torch
from augmentation import *
import os

def choose_model(args):
	if args.model == 'efficientnet':
		model = EfficientNetModel(args=args)
	elif args.model == 'convnext':
		model = ConvNextModel(args=args)
	elif args.model == 'vit':
		model = ViTModel(args=args)
	elif args.model == 'swin':
		model = SwinModel(args=args)
	elif args.model == 'cait':
		model = CaitModel(args=args)
	elif args.model == 'twins':
		model = TwinsModel(args=args)

	# Face Pretrained models
	if args.model == 'inception-resnet':
		model = InceptionResnet(args=args)
	elif args.model == 'inception-resnet-imagenet':
		model = InceptionResnetIM(args=args)

	# Part models
	if args.model == 'part-model':
		model = PartModel(args=args)
	elif args.model == 'part-model2':
		model = PartModel2(args=args)
	elif args.model == 'part-model3':
		model = PartModel3(args=args)
	elif args.model == 'eye-model':
		model = EyeModel(args=args)

	return model

def get_trainer(args, train_loader, device, optimizer, scheduler, criterion, depth_encoder=None, depth_decoder=None):
	if args.trainer == 'simple':
		trainer = SimpleTrainer(args, train_loader, device, optimizer, scheduler, criterion)
	elif args.trainer == 'part':
		trainer = PartTrainer(args, train_loader, device, optimizer, scheduler, criterion)
	return trainer

def get_transforms(args):
	if args.model in ['inception-resnet']:
		test_transforms = transforms.Compose([
		np.float32,
		transforms.ToTensor(),
		fixed_image_standardization,
		])

		if args.augmentation == 'randaugment':
			train_transforms = transforms.Compose([
				RandAugmentPC(n=2, m=args.magnitude),
				np.float32,
				transforms.ToTensor(),
				fixed_image_standardization,
			])
		elif args.augmentation == 'weak':
			train_transforms = transforms.Compose([
				transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
				np.float32,
				transforms.ToTensor(),
				fixed_image_standardization,
			])
		elif args.augmentation == 'none':
			train_transforms = transforms.Compose([
				np.float32,
				transforms.ToTensor(),
				fixed_image_standardization,
			])

	else:
		test_transforms = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225]
			)
		])
		
		if args.augmentation == 'randaugment':
			train_transforms = transforms.Compose([
				RandAugmentPC(n=2, m=args.magnitude),
				transforms.ToTensor(),
				transforms.Normalize(
					mean=[0.485, 0.456, 0.406],
					std=[0.229, 0.224, 0.225]
				)
			])
		elif args.augmentation == 'weak':
			train_transforms = transforms.Compose([
				transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
				transforms.ToTensor(),
				transforms.Normalize(
					mean=[0.485, 0.456, 0.406],
					std=[0.229, 0.224, 0.225]
				)
			])
		elif args.augmentation == 'none':
			train_transforms = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(
					mean=[0.485, 0.456, 0.406],
					std=[0.229, 0.224, 0.225]
				)
			])

	if args.trainer == 'part':
		test_transforms = transforms.Compose([
		np.float32,
		transforms.ToTensor(),
		fixed_image_standardization,
		])

		train_transforms = transforms.Compose([
				RandAugmentPC(n=2, m=args.magnitude),
				np.float32,
				transforms.ToTensor(),
				fixed_image_standardization,
			])

		eye_train_transforms = transforms.Compose([
			RandAugmentPC(n=2, m=5),
			transforms.ToTensor()
		])

		eye_test_transforms = transforms.Compose([
			transforms.ToTensor()
		])

		train_transforms = (train_transforms, eye_train_transforms)
		test_transforms = (test_transforms, eye_test_transforms)

	return train_transforms, test_transforms

def get_loaders(args, train_transforms, test_transforms):
	image_dir = os.path.join(args.data_dir, 'Image/')
	train_label_file = os.path.join(args.data_dir, 'Label/train.label')
	val_label_file = os.path.join(args.data_dir, 'Label/test.label')

	train_dataset = CustomDataset(train_label_file, image_dir, train_transforms, args.angle, args)

	train_loader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=args.batch_size, shuffle=True,
		num_workers=args.num_workers, pin_memory=True)
	
	val_dataset = CustomDataset(val_label_file, image_dir, test_transforms, args.angle, args, train=False)

	val_loader = torch.utils.data.DataLoader(
		val_dataset,
		batch_size=args.batch_size, shuffle=False,
		num_workers=args.num_workers, pin_memory=True)

	return train_dataset, train_loader, val_dataset, val_loader

def get_scheduler(args, optimizer, train_loader):
	if args.scheduler == 'step':
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
	elif args.scheduler == 'one-cycle':
		scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs)

	return scheduler
