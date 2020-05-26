#!/bin/bash

mkdir saved_models

python wrangle_KG.py FB15K237_4000
python wrangle_KG.py WN18RR_4000

cp ../pyGAT/GAT_FB15K237_4000/20200520-104343/GAT_FB15K237_4000_output.txt ./data/FB15K237_4000/FB15K237_4000.content
cp ../pyGAT/data/FB15K237_4000/FB15K237_4000.rel ./data/FB15K237_4000/FB15K237_4000.rel

cp ../pyGAT/GAT_WN18RR_4000/20200520-195544/GAT_WN18RR_4000_output.txt ./data/WN18RR_4000/WN18RR_4000.content
cp ../pyGAT/data/WN18RR_4000/WN18RR_4000.rel ./data/WN18RR_4000/WN18RR_4000.rel


cp ../pyGAT/GAT_FB15K237_4000_rel/20200520-104351/GAT_FB15K237_4000_output.txt ./data/FB15K237_4000_rel/FB15K237_4000_rel.content
cp ../pyGAT/data/FB15K237_4000/FB15K237_4000.rel ./data/FB15K237_4000_rel/FB15K237_4000_rel.rel
cp ./data/FB15K237_4000/*.json ./data/FB15K237_4000_rel/

cp ../pyGAT/GAT_WN18RR_4000_rel/20200525-234900/GAT_WN18RR_4000_output.txt ./data/WN18RR_4000_rel/WN18RR_4000_rel.content
cp ../pyGAT/data/WN18RR_4000/WN18RR_4000.rel ./data/WN18RR_4000_rel/WN18RR_4000_rel.rel
cp ./data/WN18RR_4000/*.json ./data/WN18RR_4000_rel/


cp ../pyGAT/GAT_FB15K237_4000_adsf/20200525-222338/GAT_FB15K237_4000_output.txt ./data/FB15K237_4000_adsf/FB15K237_4000_adsf.content
cp ../pyGAT/data/FB15K237_4000/FB15K237_4000.rel ./data/FB15K237_4000_adsf/FB15K237_4000_adsf.rel
cp ./data/FB15K237_4000/*.json ./data/FB15K237_4000_adsf/

cp ../pyGAT/GAT_WN18RR_4000_adsf/20200525-234519/GAT_WN18RR_4000_output.txt ./data/WN18RR_4000_adsf/WN18RR_4000_adsf.content
cp ../pyGAT/data/WN18RR_4000/WN18RR_4000.rel ./data/WN18RR_4000_adsf/WN18RR_4000_adsf.rel
cp ./data/WN18RR_4000/*.json ./data/WN18RR_4000_adsf/


cp ../pyGAT/GAT_FB15K237_4000_all/20200520-104745/GAT_FB15K237_4000_output.txt ./data/FB15K237_4000_all/FB15K237_4000_all.content
cp ../pyGAT/data/FB15K237_4000/FB15K237_4000.rel ./data/FB15K237_4000_all/FB15K237_4000_all.rel
cp ./data/FB15K237_4000/*.json ./data/FB15K237_4000_all/

cp ../pyGAT/GAT_WN18RR_4000_all/20200520-104941/GAT_WN18RR_4000_output.txt ./data/WN18RR_4000_all/WN18RR_4000_all.content
cp ../pyGAT/data/WN18RR_4000/WN18RR_4000.rel ./data/WN18RR_4000_all/WN18RR_4000_all.rel
cp ./data/WN18RR_4000/*.json ./data/WN18RR_4000_all/
