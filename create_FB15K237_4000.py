import numpy as np


dateset_ori = "FB15K237"
dateset_new = dateset_ori + "_4000"

files = ["train.txt", "valid.txt", "test.txt"]

entities_names = np.genfromtxt("../pyGAT/data/{}/{}.content".format(dateset_new, dateset_new), dtype=np.dtype(str))[:, 0]
entities_names = set(entities_names)
relations_names = np.genfromtxt("../pyGAT/data/{}/{}.rel".format(dateset_new, dateset_new), dtype=np.dtype(str))[:, 0]
relations_names = set(relations_names)

for file in files:
    file_path = "./data/" + dateset_ori + '/' + file
    output_f = open("./data/" + dateset_new + '/' + file, 'w')
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            e1, rel, e2 = line[:-1].split('\t')
            if e1 in entities_names and e2 in entities_names and rel in relations_names:
                output_f.write(line)
    output_f.close()
