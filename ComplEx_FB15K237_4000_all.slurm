#!/bin/bash
#SBATCH -o ./ComplEx_FB15K237_4000_all/ComplEx_FB15K237_4000_result_%j.log         # 输出日志，可以查看ckpt文件夹
#SBATCH -J CmF4al             # 任务名称
#SBATCH --gres=gpu:V100:1
#SBATCH -c 5
#SBATCH -p sugon
#SBATCH -x sugon-gpu-5,sugon-gpu-6
time=`date +%Y%m%d-%H%M%S`
mkdir ./ComplEx_FB15K237_4000_all/${time}
source /home/LAB/anaconda3/etc/profile.d/conda.sh
conda activate cuda9.1
CUDA_VISIBLE_DEVICES=0 python main.py model ComplEx dataset FB15K237_4000_all \
                                      input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 \
                                      lr 0.003 lr_decay 0.995 process True
mv ./ComplEx_FB15K237_4000_all/ComplEx_FB15K237_4000_result_$SLURM_JOB_ID.log ./ComplEx_FB15K237_4000_all/${time}/ComplEx_FB15K237_4000_result.log
