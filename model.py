import torch
from torch.nn import functional as F, Parameter
from torch.autograd import Variable


from spodernet.utils.global_config import Config
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Complex(torch.nn.Module):
    def __init__(self, num_entities, num_relations, emb_e_real, emb_e_img):
        super(Complex, self).__init__()
        self.num_entities = num_entities
        self.emb_e_real = emb_e_real
        self.emb_e_img = emb_e_img
        self.emb_rel_real = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self, init_emb_rel):
        # 初始化为正态分布结果
        # xavier_normal_(self.emb_e_real.weight.data)
        # xavier_normal_(self.emb_e_img.weight.data)
        # xavier_normal_(self.emb_rel_real.weight.data)
        # xavier_normal_(self.emb_rel_img.weight.data)
        # 初始化为GAT的结果
        self.emb_rel_real.weight.data.copy_(torch.from_numpy(init_emb_rel))
        self.emb_rel_img.weight.data.copy_(torch.from_numpy(init_emb_rel))

    def forward(self, e1, rel):
        emb_e_real = self.emb_e_real.cuda()
        emb_e_img = self.emb_e_img.cuda()

        e1_embedded_real = emb_e_real[e1].squeeze()
        rel_embedded_real = self.emb_rel_real(rel).squeeze()
        e1_embedded_img = emb_e_img[e1].squeeze()
        rel_embedded_img = self.emb_rel_img(rel).squeeze()

        e1_embedded_real = self.inp_drop(e1_embedded_real)
        rel_embedded_real = self.inp_drop(rel_embedded_real)
        e1_embedded_img = self.inp_drop(e1_embedded_img)
        rel_embedded_img = self.inp_drop(rel_embedded_img)

        # complex space bilinear product (equivalent to HolE)
        realrealreal = torch.mm(e1_embedded_real*rel_embedded_real, emb_e_real.transpose(1, 0))
        realimgimg = torch.mm(e1_embedded_real*rel_embedded_img, emb_e_img.transpose(1, 0))
        imgrealimg = torch.mm(e1_embedded_img*rel_embedded_real, emb_e_img.transpose(1, 0))
        imgimgreal = torch.mm(e1_embedded_img*rel_embedded_img, emb_e_real.transpose(1, 0))
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = F.sigmoid(pred)

        return pred


class DistMult(torch.nn.Module):
    def __init__(self, num_entities, num_relations, emb_e):
        super(DistMult, self).__init__()
        self.emb_e = emb_e
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)

        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self, init_emb_rel):
        # 初始化为正态分布结果
        # xavier_normal_(self.emb_e.weight.data)
        # xavier_normal_(self.emb_rel.weight.data)
        # 初始化为GAT的结果
        self.emb_rel.weight.data.copy_(torch.from_numpy(init_emb_rel))

    def forward(self, e1, rel):
        # e1: batch_size * 1, rel: batch_size * 1
        emb_e = self.emb_e.cuda()
        e1_embedded = emb_e[e1]  # batch_size * 1 * embedding_e_dim
        rel_embedded = self.emb_rel(rel)  # batch_size * 1 * embedding_rel_dim
        e1_embedded = e1_embedded.squeeze()  # batch_size * embedding_e_dim
        rel_embedded = rel_embedded.squeeze()  # batch_size * embedding_rel_dim

        e1_embedded = self.inp_drop(e1_embedded)
        rel_embedded = self.inp_drop(rel_embedded)

        pred = torch.mm(e1_embedded * rel_embedded, emb_e.transpose(1, 0))  # batch_size * num_entities
        pred = F.sigmoid(pred)

        return pred


class ConvE(torch.nn.Module):
    def __init__(self, num_entities, num_relations, emb_e):
        super(ConvE, self).__init__()
        self.emb_e = emb_e
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)

        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout)
        self.feature_map_drop = torch.nn.Dropout2d(Config.feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=Config.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(4608, Config.embedding_dim)

    def init(self, init_emb_rel):
        # 初始化为正态分布结果
        # xavier_normal_(self.emb_e.weight.data)
        # xavier_normal_(self.emb_rel.weight.data)
        # 初始化为GAT的结果
        self.emb_rel.weight.data.copy_(torch.from_numpy(init_emb_rel))

    def forward(self, e1, rel):
        emb_e = self.emb_e.cuda()
        e1_embedded = emb_e[e1].view(-1, 1, 10, 10)  # batch_size * 1 * 10 * 10
        rel_embedded = self.emb_rel(rel).view(-1, 1, 10, 10)  # batch_size * 1 * 10 * 10

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)  # batch_size * 1 * 20 * 20

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)  # batch_size * 32 * 18 * 8
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(Config.batch_size, -1)  # batch_size * (32 * 18 * 8)
        # print(x.size())
        x = self.fc(x)  # batch_size * 100
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, emb_e.transpose(1, 0))  # batch_size * N
        x += self.b.expand_as(x)
        pred = F.sigmoid(x)

        return pred
