# ILA-plus-plus
Code for our paper [Deepfake Forensics via An Adversarial Game](https://arxiv.org/abs/2103.13567).

## Requirements
* Python 3.7
* Numpy 1.19.5
* Pillow 8.4.0
* PyTorch 1.8.1
* Torchvision 0.9.1
* imageio 2.13.5
* opencv-python 4.5.5.64
* TensorboardX 2.4.1
* Albumentations 1.1.0

## Datasets
FaceForensics++ (FF++) is a recently released large scale deepfake video detection dataset, containing 1,000 real videos, in which 720 videos were used for training, 140 videos were reserved for verification, and 140 videos were used for test. Each real video in the dataset was manipulated using four advanced methods, including DeepFakes (DF), Face2Face (F2F),FaceSwap (FS), and NeuralTextures (NT), to generate four fake videos. We followed the official split of the training, validation, and test sets in our experiments. Each video in the dataset were processed to have three video qualities, namely RAW, C23 , and C40 . For each quality, there are 5,000 (real and fake) videos in total, and we extracted 270 frames from each video following the official implementation of face detection and alignment in. 

To make the evaluation more comprehensive, we introduced two more deepfake datasets: DFD and Celeb-DF. DFD contains 3,068 deepfake videos, which were forged based on 363 real videos. Celeb-DF contains 590 real videos and 5,639 fake videos.

## training
Training with generators:
```
python detect_gen.py \
    --exp_name gen_exp \
    --max_epoch 120 \
    --batch_size 32 \
    --snapshot_interval 5 \
    --lr 5e-4 \
    --lr_decay_epochs 15 \
    --gen_train_type default \
    --g_per_d 3
```

Training without generators:
```
python detect_grad.py \
    --exp_name grad_exp \
    --max_epoch 120 \
    --batch_size 32 \
    --snapshot_interval 5 \
    --lr 5e-4 \
    --lr_decay_epochs 15 \
    --gen_train_type default \
    --g_per_d 1
```


## Evaluation
```
python eval.py --exp_name $exp_name --epoch $i --multi --model_type b7
```

## Usage
Full usage:
```
python detect_grad.py -h
usage: detect_grad.py [-h] [--mean] [--weight] [--exp_name EXP_NAME] [--model_type MODEL_TYPE]
               [--pretrain] [--pretrain_name PRETRAIN_NAME] [--dataset DATASET]
               [--dropout_rate DROPOUT_RATE] [--dataset_type DATASET_TYPE]
               [--val_dataset_type VAL_DATASET_TYPE] [--stride STRIDE] [--fc_model] [--multi]       
               [--fake_type {FaceSwap,NeuralTextures,Face2Face,Deepfakes}] [--cmp_level CMP_LEVEL]  
               [--all] [--no_clamp] [--norm] [--norm_value NORM_VALUE] [--learn_norm] [--sigmoid]   
               [--tanh] [--clamp_res] [--fake_set] [--real_set] [--no_res] [--aug_before]
               [--aug_before_full] [--aug_after] [--perm] [--aug_blur] [--check_blur] [--bn2]       
               [--kernel_channels KERNEL_CHANNELS] [--radius RADIUS] [--pretrain_g]
               [--gen_norm_type GEN_NORM_TYPE] [--gen_train_type GEN_TRAIN_TYPE]
               [--g_per_d G_PER_D] [--gen_blocks GEN_BLOCKS] [--gen_lr GEN_LR]
               [--max_epoch MAX_EPOCH] [--batch_size BATCH_SIZE] [--image_size IMAGE_SIZE]
               [--lr LR] [--lr_decay_factor LR_DECAY_FACTOR] [--lr_decay_epochs LR_DECAY_EPOCHS]    
               [--weight_decay WEIGHT_DECAY] [--momentum MOMENTUM]
               [--snapshot_interval SNAPSHOT_INTERVAL] [--log_interval LOG_INTERVAL]
               [--hist_interval HIST_INTERVAL] [--segment_length SEGMENT_LENGTH] [--workers N]      

optional arguments:
  -h, --help            show this help message and exit
  --mean                Train single-frame model with 5 frame mean
  --weight              Use weighted version loss
  --exp_name EXP_NAME   name of experiment
  --model_type MODEL_TYPE
                        model type for backbone
  --pretrain            Use our pretrained model
  --pretrain_name PRETRAIN_NAME
                        model type for backbone
  --dataset DATASET     Dataset type
  --dropout_rate DROPOUT_RATE
                        Dropout rate for cls model
  --dataset_type DATASET_TYPE
                        Dataset type
  --val_dataset_type VAL_DATASET_TYPE
                        Dataset type
  --stride STRIDE       stride for b5 model.
  --fc_model            Use fc model to classify
  --multi               Use multi types of fake
  --fake_type {FaceSwap,NeuralTextures,Face2Face,Deepfakes}
                        Fake type
  --cmp_level CMP_LEVEL
                        Cmp level
  --all                 Use more general kernel
  --no_clamp            Not clip sigma
  --norm                Norm sigma
  --norm_value NORM_VALUE
                        Norm value for stadv.
  --learn_norm          Use learnable norm
  --sigmoid             Use sigmoid other than abs as sigma
  --tanh                Use tanh for flow generator
  --clamp_res           Clamp res image
  --fake_set            Use fake_set
  --real_set            Use real_set
  --no_res              Dataset gen not use res
  --aug_before          Use transform before gen
  --aug_before_full     Use full transform before gen
  --aug_after           Use transform before gen
  --perm                Perm inputs when training dis
  --aug_blur            Aug blur image for a fair compair
  --check_blur          Use sigmoid other than abs as sigma
  --bn2                 Use multi bn for real and fake images
  --kernel_channels KERNEL_CHANNELS
                        channel num for kernels gen model .
  --radius RADIUS       radius of gaussian kernel.
  --pretrain_g          Use our pretrained model
  --gen_norm_type GEN_NORM_TYPE
                        type of norm in model-gen
  --gen_train_type GEN_TRAIN_TYPE
                        Use lsgan-like loss for g or wgan-like loss
  --g_per_d G_PER_D     # of train g every d
  --gen_blocks GEN_BLOCKS
                        # of res blocks in generator
  --gen_lr GEN_LR       Initial model gen learning rate.
  --max_epoch MAX_EPOCH
                        Number of training epoches.
  --batch_size BATCH_SIZE
                        Number of samples per minibatch for training.
  --image_size IMAGE_SIZE
                        Width and hight of trainng images.
  --lr LR               Initial learning rate.
  --lr_decay_factor LR_DECAY_FACTOR
                        The factor to decay learning rate.
  --lr_decay_epochs LR_DECAY_EPOCHS
                        Every 200 epochs to decay the learning rate.
  --weight_decay WEIGHT_DECAY
                        Parameter for weigth decay.
  --momentum MOMENTUM   Solver momentum
  --snapshot_interval SNAPSHOT_INTERVAL
                        Snapshot per X epochs during training.
  --log_interval LOG_INTERVAL
                        Log per X epochs during training.
  --hist_interval HIST_INTERVAL
                        Log per X epochs during training.
  --segment_length SEGMENT_LENGTH
                        Length of video segment to be used.
  --workers N           Number of data loading workers (default: 4)
```

## Acknowledgements
The following resources are very helpful for our work:

* [FaceForensics++ Dataset](https://github.com/ondyari/FaceForensics)
* [Celeb-DF Dataset](https://github.com/yuezunli/celeb-deepfakeforensics)
* [GAN](https://arxiv.org/abs/1406.2661)
* [CycleGA](https://junyanz.github.io/CycleGAN)
* [RAdam](https://github.com/LiyuanLucasLiu/RAdam)
* [Apex](https://github.com/NVIDIA/apex)


## Citation
Please cite our work in your publications if it helps your research:

```
@article{wang2022deepfake,
  title={Deepfake Forensics via An Adversarial Game},
  author={Wang, Zhi and Guo, Yiwen and Zuo, Wangmeng},
  journal={IEEE Transactions on Image Processing},
  year={2022},
  publisher={IEEE}
}
```
