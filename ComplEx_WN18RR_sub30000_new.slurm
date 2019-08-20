#!/bin/bash
#SBATCH -o ./ComplEx_WN18RR_sub30000_new/ComplEx_WN18RR_sub30000_new_result.log         # 输出日志，可以查看ckpt文件夹
#SBATCH -J CmWs3n             # 任务名称
#SBATCH --gres=gpu:V100:1
#SBATCH -c 5
source /home/LAB/anaconda3/etc/profile.d/conda.sh
conda activate cuda9.1
CUDA_VISIBLE_DEVICES=0 python main.py model ComplEx dataset WN18RR_sub30000_new \
                                      input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 \
                                      lr 0.003 lr_decay 0.995 process True
