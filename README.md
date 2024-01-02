# Run configuration

Sample experiments:
CIFAR-100 (observing one batch at a time)

Pre-trained representation can be obtained at https://github.com/VICO-UoE/URL, specifically https://drive.google.com/file/d/1Dv8TX6iQ-BE2NMpfd0sQmH2q4mShmo1A/view

```
python main.py --model scroll --dataset seq-cifar100 --buffer_size 500 --load_best_args
python main.py --model scroll --dataset seq-cifar100 --buffer_size 2000 --load_best_args
```

Mini-ImageNet (base task is 64 classes, with remaining 36 classes observed 2 at a time.)
1. Generate the mini_imagenet experiment setting with mini_imagenet_100_classes_dataset_generator.py. The original mini_imagenet data is obtained from https://www.dropbox.com/sh/6yd1ygtyc3yd981/AABVeEqzC08YQv4UZk7lNHvya?dl=0.
2. The script below skips observing the sequence explicitly due to schedule-invariance. Both replay buffer and f_T is deterministic in this case.
```
python srcoll_mini.py
```
