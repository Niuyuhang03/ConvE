#!/bin/bash
# mkdir data
# mkdir data/FB15k-237
# mkdir data/WN18RR
mkdir saved_models

# tar -xvf WN18RR.tar.gz -C data/WN18RR
# tar -xvf FB15k-237.tar.gz -C data/FB15k-237

python wrangle_KG.py FB15k-237
python wrangle_KG.py FB15k-237_rel
python wrangle_KG.py WN18RR
python wrangle_KG.py WN18RR_rel
python wrangle_KG.py WN18RR_sub30000
python wrangle_KG.py WN18RR_sub30000_rel