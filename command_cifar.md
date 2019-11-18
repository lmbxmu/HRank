



********************************************************************************************************
vgg-16-bn
********************************************************************************************************
python rank_generation.py \
--resume ./result/pretrain/vgg_16_bn.pt \
--arch vgg_16_bn \
--limit 1 \
--compress_rate [0.0]*13 \
--gpu 0


python main.py \
--job_dir ./result/vgg_16_bn/test \
--resume ./result/pretrain/vgg_16_bn.pt \
--epoch 1 \
--adjust_prune_ckpt \
--arch vgg_16_bn \
--compress_rate [0.3]*7+[0.7]*6 \
--gpu 0


********************************************************************************************************
resnet_56_convwise
********************************************************************************************************

python my_rank_generation.py \
--resume result/resnet_56_convwise/pruned_checkpoint/resnet_56.pt \
--arch resnet_56_convwise \
--limit 5 \
--compress_rate [0.1]+[0.70]*18+[0.70]*18+[0.70]*18 \
--gpu 3


python main.py \
--job_dir result/resnet_56/test1 \
--resume ./result/pretrain/resnet_56.pt \
--epochs 1 \
--adjust_prune_ckpt \
--arch resnet_56 \
--compress_rate [0.0]+[0.50]*18+[0.10]+[0.20]*17+[0.1]+[0.20]*5+[0.1]*12 \
--gpu 0



********************************************************************************************************
densenet_40
********************************************************************************************************
python my_rank_generation.py \
--resume /media/disk2/zyc/prune_result/densenet_40/pruned_checkpoint/densenet_40.pt \
--arch densenet_40 \
--limit 1 \
--compress_rate [0.]+[0.90]*10+[0.8]*2+[0.60]+[0.8]+[0.9]*11+[0.50]+[0.7]+[0.9]*8+[0.8]+[0.9]+[0.5] \
--gpu 0



python main.py \
--job_dir result/densenet_40/test \
--resume ./result/pretrain/densenet_40.pt \
--epochs 1 \
--adjust_prune_ckpt \
--arch densenet_40 \
--compress_rate [0.0]+[0.1]*6+[0.6]*6+[0.2]+[0.1]*6+[0.6]*6+[0.2]+[0.1]*6+[0.6]*5+[0.0] \
--gpu 0




********************************************************************************************************
googlenet
********************************************************************************************************

python my_rank_generation.py \
--resume /media/disk2/zyc/prune_result/googlenet/pruned_checkpoint/googlenet.pt \
--arch googlenet \
--limit 100 \
--start_idx 8 \
--gpu 0

python main.py \
--job_dir result/googlenet/test \
--resume ./result/pretrain/googlenet.pt \
--epochs 1 \
--compress_rate [0.10]+[0.4]*2+[0.4]*3+[0.5]*2+[0.5]*2 \
--adjust_prune_ckpt \
--arch googlenet \
--gpu 0,1




********************************************************************************************************
resnet_110
********************************************************************************************************
python my_rank_generation.py \
--resume /media/disk2/zyc/prune_result/resnet_110/pruned_checkpoint/resnet_110.pt \
--arch resnet_110_convwise \
--limit 10 \
--compress_rate [0.1]+[0.70]*36+[0.70]*36+[0.70]*36 \
--gpu 0


python main.py \
--job_dir result/resnet_110/test \
--resume ./result/pretrain/resnet_110.pt \
--epochs 1 \
--adjust_prune_ckpt \
--arch resnet_110 \
--compress_rate [0.1]+[0.40]*36+[0.40]*36+[0.4]*36 \
--gpu 0


********************************************************************************************************
resnet_50
********************************************************************************************************

CUDA_VISIBLE_DEVICES=0,1
python main.py \
--dataset imagenet \
--data_dir /media/disk2/zyc/ImageNet2012 \
--job_dir ./result/resnet_50/test \
--resume ./result/pretrain/resnet50-19c8e357.pth \
--epochs 30 \
--lr 0.001 \
--lr_decay_step 10,20 \
--adjust_prune_ckpt \
--train_batch_size 32 \
--eval_batch_size 32 \
--arch resnet_50 \
--compress_rate [0.2]+[0.80]*10+[0.80]*13+[0.80]*19+[0.70]*10 \
--gpu 0,1