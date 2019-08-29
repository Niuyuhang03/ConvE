#!/bin/bash

mkdir saved_models

python wrangle_KG.py FB15k-237_origin
python wrangle_KG.py FB15k-237
python wrangle_KG.py FB15k-237_rel
python wrangle_KG.py WN18RR_origin
python wrangle_KG.py WN18RR
python wrangle_KG.py WN18RR_rel
python wrangle_KG.py WN18RR_sub30000
python wrangle_KG.py WN18RR_sub30000_rel

myFile="../pyGAT/GAT_result/GAT_FB15K237_mean/GAT_FB15K237_output.txt"
if [ -f "$myFile" ];then
	cp $myFile ./data/FB15k-237/FB15k-237.content
else
	echo "$myFile doesnt exist"
fi

myFile="../pyGAT/GAT_result/GAT_FB15K237_rel_mean/GAT_FB15K237_output.txt"
if [ -f "$myFile" ];then
	cp $myFile ./data/FB15k-237_rel/FB15k-237_rel.content
else
        echo "$myFile doesnt exist"
fi

myFile="../pyGAT/GAT_result/GAT_WN18RR_mean/GAT_WN18RR_output.txt"
if [ -f "$myFile" ];then
        cp $myFile ./data/WN18RR/WN18RR.content
else
        echo "$myFile doesnt exist"
fi

myFile="../pyGAT/GAT_result/GAT_WN18RR_rel_mean/GAT_WN18RR_output.txt"
if [ -f "$myFile" ];then
        cp $myFile ./data/WN18RR_rel/WN18RR_rel.content
else
        echo "$myFile doesnt exist"
fi

myFile="../pyGAT/GAT_result/GAT_WN18RR_sub30000_mean/GAT_WN18RR_sub30000_output.txt"
if [ -f "$myFile" ];then
	cp $myFile ./data/WN18RR_sub30000/WN18RR_sub30000.content
else
	echo "$myFile doesnt exist"
fi

myFile="../pyGAT/GAT_result/GAT_WN18RR_sub30000_rel_mean/GAT_WN18RR_sub30000_output.txt"
if [ -f "$myFile" ];then
        cp $myFile ./data/WN18RR_sub30000_rel/WN18RR_sub30000_rel.content
else
        echo "$myFile doesnt exist"
fi

