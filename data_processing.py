import numpy as np


# (embedding数据集, 要被删除的原始数据集, 删除的结果数据集)的元组
dataset_list = [('FB15K237', 'FB15k-237_origin', 'FB15k-237'),
                ('FB15K237', 'FB15k-237_origin', 'FB15k-237_rel'),
                ('WN18RR', 'WN18RR_origin', 'WN18RR'),
                ('WN18RR', 'WN18RR_origin', 'WN18RR_rel'),
                ('WN18RR_sub30000', 'WN18RR_origin', 'WN18RR_sub30000'),
                ('WN18RR_sub30000', 'WN18RR_origin', 'WN18RR_sub30000_rel')]

# 删除不在.content和.rel中出现过的ConvE三元组
for dataset_triple in dataset_list:
    dateset1, dataset2, dataset3 = dataset_triple
    content_path = '../pyGAT/data/{}/{}.content'.format(dateset1, dateset1)
    rel_path = '../pyGAT/data/{}/{}.rel'.format(dateset1, dateset1)
    origin_path = ['./data/{}/train.txt'.format(dataset2), './data/{}/valid.txt'.format(dataset2), './data/{}/test.txt'.format(dataset2)]
    output_path = ['./data/{}/train.txt'.format(dataset3), './data/{}/valid.txt'.format(dataset3), './data/{}/test.txt'.format(dataset3)]

    entity_name = list(np.genfromtxt(content_path, dtype=np.dtype(str))[:, 0])
    rel_name = list(np.genfromtxt(rel_path, dtype=np.dtype(str))[:, 0])

    for index in range(len(output_path)):
        num = 0
        with open(origin_path[index], 'r') as f1:
            lines = f1.readlines()
            all = len(lines)
            with open(output_path[index], 'w') as f2:
                for line in lines:
                    e1, rel, e2 = line.replace('\n','').split('\t')
                    if e1 not in entity_name or e2 not in entity_name or rel not in rel_name:
                        num += 1
                    else:
                        f2.write(line)
        print('{} delete {}/{} triples'.format(output_path[index], num, all))
