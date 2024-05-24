#/bin/bash

# No fusion
python main.py --model inception-resnet --epochs 200 --exp_index 3 --loss angular --batch_size 256 --lr 1e-4 --magnitude 3 --run_name rand-init-no-fusion --trainer simple --augmentation randaugment --seed 0
python main.py --model inception-resnet --epochs 200 --exp_index 3 --loss angular --batch_size 256 --lr 1e-4 --magnitude 3 --run_name pretrained-no-fusion --trainer simple --augmentation randaugment --pretrained True --seed 0

# FCN
python main.py --model part-model --epochs 200 --exp_index 3 --loss angular --batch_size 256 --lr 1e-4 --magnitude 3 --run_name rand-init-multimodal-fcn --trainer part --augmentation randaugment --seed 0
python main.py --model part-model --epochs 200 --exp_index 3 --loss angular --batch_size 256 --lr 1e-4 --magnitude 3 --run_name pretrained-multimodal-fcn --trainer part --augmentation randaugment --pretrained True --seed 0

# Cross Attention
python main.py --model part-model3 --epochs 200 --exp_index 3 --loss angular --batch_size 256 --lr 1e-4 --magnitude 3 --run_name rand-rand-init-multimodal-crossatention --trainer part --augmentation randaugment --seed 0
python main.py --model part-model3 --epochs 200 --exp_index 3 --loss angular --batch_size 256 --lr 1e-4 --magnitude 3 --run_name pretrained-multimodal-crossatention --trainer part --augmentation randaugment --pretrained True --seed 0

