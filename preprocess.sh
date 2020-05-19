#!/bin/bash

mkdir saved_models

python wrangle_KG.py FB15K237_4000
python wrangle_KG.py WN18RR_4000

cp ../pyGAT/GAT_FB15K237_4000/20200519-124400/GAT_FB15K237_4000_output.txt ./data/FB15K237_4000/FB15K237_4000.content
cp ../pyGAT/data/FB15K237_4000/FB15K237_4000.rel ./data/FB15K237_4000/FB15K237_4000.rel

cp ../pyGAT/GAT_WN18RR_4000/20200519-122342/GAT_WN18RR_4000_output.txt ./data/WN18RR_4000/WN18RR_4000.content
cp ../pyGAT/data/WN18RR_4000/WN18RR_4000.rel ./data/WN18RR_4000/WN18RR_4000.rel


cp ../pyGAT/GAT_FB15K237_4000_rel//GAT_FB15K237_4000_output.txt ./data/FB15K237_4000_rel/FB15K237_4000_rel.content
cp ../pyGAT/data/FB15K237_4000/FB15K237_4000.rel ./data/FB15K237_4000_rel/FB15K237_4000_rel.rel

cp ../pyGAT/GAT_WN18RR_4000_rel//GAT_WN18RR_4000_output.txt ./data/WN18RR_4000_rel/WN18RR_4000_rel.content
cp ../pyGAT/data/WN18RR_4000/WN18RR_4000.rel ./data/WN18RR_4000_rel/WN18RR_4000_rel.rel


cp ../pyGAT/GAT_FB15K237_4000_adsf//GAT_FB15K237_4000_output.txt ./data/FB15K237_4000_adsf/FB15K237_4000_adsf.content
cp ../pyGAT/data/FB15K237_4000/FB15K237_4000.rel ./data/FB15K237_4000_adsf/FB15K237_4000_rel.rel

cp ../pyGAT/GAT_WN18RR_4000_adsf//GAT_WN18RR_4000_output.txt ./data/WN18RR_4000_adsf/WN18RR_4000_adsf.content
cp ../pyGAT/data/WN18RR_4000/WN18RR_4000.rel ./data/WN18RR_4000_adsf/WN18RR_4000_adsf.rel


cp ../pyGAT/GAT_FB15K237_4000_all//GAT_FB15K237_4000_output.txt ./data/FB15K237_4000_all/FB15K237_4000_all.content
cp ../pyGAT/data/FB15K237_4000/FB15K237_4000.rel ./data/FB15K237_4000_all/FB15K237_4000_rel.rel

cp ../pyGAT/GAT_WN18RR_4000_all//GAT_WN18RR_4000_output.txt ./data/WN18RR_4000_all/WN18RR_4000_all.content
cp ../pyGAT/data/WN18RR_4000/WN18RR_4000.rel ./data/WN18RR_4000_all/WN18RR_4000_all.rel
