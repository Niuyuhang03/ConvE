#!/bin/bash
#SBATCH -o ./DistMult_FB15K237_4000/DistMult_FB15K237_4000_result_%j.log         # 输出日志，可以查看ckpt文件夹
#SBATCH -J DiF4             # 任务名称
#SBATCH --gres=gpu:V100:1
#SBATCH -c 5
#SBATCH -p sugon
time=`date +%Y%m%d-%H%M%S`
mkdir ./DistMult_FB15K237_4000/${time}
source /home/LAB/anaconda3/etc/profile.d/conda.sh
conda activate cuda9.1
CUDA_VISIBLE_DEVICES=0 python main.py model DistMult dataset FB15K237_4000 \
                                      input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 \
                                      lr 0.003 lr_decay 0.995 process True
mv ./DistMult_FB15K237_4000/DistMult_FB15K237_4000_result_$SLURM_JOB_ID.log ./DistMult_FB15K237_4000/${time}/DistMult_FB15K237_4000_result.log
