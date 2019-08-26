import numpy as np
import scipy.sparse as sp
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="WN18RR",
                help="Dataset string ('aifb', 'mutag', 'bgs', 'am')")

ap.add_argument("-f", "--file", type=str, default="WN18RR",
                help="Dataset string ('aifb', 'mutag', 'bgs', 'am')")

args = vars(ap.parse_args())

print(args)

# Define parameters
dataset = args['dataset'] # 对比数据集
file = args['file'] # 需要删除的数据集
path1 = '../../../pyGAT/data/' + dataset + '/'
path2 = './data/' + file + '/'
file_name = 'train'

name = []
rel_name = []
entity_1 = []
rel = []
entity_2 = []
index = 0
num = 0

print('Loading {} dataset...'.format(dataset))
idx_features_labels = np.genfromtxt("{}{}.content".format(path1, dataset), dtype=np.dtype(str))
idx_rel = np.genfromtxt("{}{}.rel".format(path1, dataset), dtype=np.dtype(str))

entity_rel_entity = np.genfromtxt("{}{}.txt".format(path2, file_name), dtype=np.dtype(str))
with open(path2 + file_name + '.txt', 'r') as f:
    lines = f.readlines()

entity_1 = entity_rel_entity[: , 0]
rel = entity_rel_entity[: , 1]
entity_2 = entity_rel_entity[: , -1]

name = idx_features_labels[: , 0]
rel_name = idx_rel[: , 0]

d1 = {}
d2 = {}
for i in range(len(name)):
    d1[name[i]] = 1

for i in range(len(rel_name)):
    d2[rel_name[i]] = 1

## entity
with open(path2 + file_name + '.txt', 'w') as f:
    for i in range(len(entity_1)):
        index = 0
        if(entity_1[i] in d1 and entity_2[i] in d1):
            index = 1
            f.write(lines[i])
        if(index == 0):
            print(i)
            num = num + 1

# # rel
# with open(path2 + file_name + '.txt', 'w') as f:
#     for i in range(len(rel)):
#         index = 0
#         if(rel[i] in d2):
#             index = 1
#             f.write(lines[i])
#         if(index == 0):
#             print(i)
#             num = num + 1

f.close()
print('Done! The useless data num is: ', num)