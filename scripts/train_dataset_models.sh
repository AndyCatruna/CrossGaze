#/bin/bash

python main.py --model inception-resnet-imagenet --epochs 200 --exp_index 2 --loss angular --batch_size 256 --lr 1e-4 --magnitude 3 --run_name imagenet-inception-resnet --trainer simple --augmentation randaugment --pretrained True --seed 0
python main.py --model inception-resnet --epochs 200 --exp_index 2 --loss angular --batch_size 256 --lr 1e-4 --magnitude 3 --run_name casia-webface-inception-resnet --trainer simple --augmentation randaugment --pretrained True --seed 0 --pretraining_dataset casia-webface
python main.py --model inception-resnet --epochs 200 --exp_index 2 --loss angular --batch_size 256 --lr 1e-4 --magnitude 3 --run_name vggface2-inception-resnet --trainer simple --augmentation randaugment --pretrained True --seed 0 --pretraining_dataset vggface2

