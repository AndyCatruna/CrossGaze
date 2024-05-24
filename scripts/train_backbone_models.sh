#/bin/bash

python main.py --model efficientnet --epochs 200 --exp_index 1 --loss angular --batch_size 256 --lr 1e-4 --magnitude 3 --run_name pretrained-efficientnet --trainer simple --augmentation randaugment --pretrained True --seed 0
python main.py --model efficientnet --epochs 200 --exp_index 1 --loss angular --batch_size 256 --lr 1e-4 --magnitude 3 --run_name rand-init-efficientnet --trainer simple --augmentation randaugment --seed 0

python main.py --model swin --epochs 200 --exp_index 1 --loss angular --batch_size 256 --lr 1e-4 --magnitude 3 --run_name pretrained-swin --trainer simple --augmentation randaugment --pretrained True --seed 0
python main.py --model swin --epochs 200 --exp_index 1 --loss angular --batch_size 256 --lr 1e-4 --magnitude 3 --run_name rand-init-swin --trainer simple --augmentation randaugment --seed 0

python main.py --model convnext --epochs 200 --exp_index 1 --loss angular --batch_size 256 --lr 1e-4 --magnitude 3 --run_name pretrained-convnext --trainer simple --augmentation randaugment --pretrained True --seed 0
python main.py --model convnext --epochs 200 --exp_index 1 --loss angular --batch_size 256 --lr 1e-4 --magnitude 3 --run_name rand-init-convnext --trainer simple --augmentation randaugment --seed 0

python main.py --model inception-resnet --epochs 200 --exp_index 1 --loss angular --batch_size 256 --lr 1e-4 --magnitude 3 --run_name rand-init-inception-resnet --trainer simple --augmentation randaugment --seed 0
python main.py --model inception-resnet-imagenet --epochs 200 --exp_index 1 --loss angular --batch_size 256 --lr 1e-4 --magnitude 3 --run_name pretrained-inception-resnet --trainer simple --augmentation randaugment --pretrained True --seed 0
