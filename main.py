import json
import time
import torch
import pickle
import numpy as np
import argparse
import sys
import os
import math

from os.path import join
import torch.backends.cudnn as cudnn

from evaluation import ranking_and_hits
from model import ConvE, DistMult, Complex

from spodernet.preprocessing.pipeline import Pipeline, DatasetStreamer
from spodernet.preprocessing.processors import JsonLoaderProcessors, Tokenizer, AddToVocab, SaveLengthsToState, StreamToHDF5, SaveMaxLengthsToState, CustomTokenizer
from spodernet.preprocessing.processors import ConvertTokenToIdx, ApplyFunction, ToLower, DictKey2ListMapper, ApplyFunction, StreamToBatch
from spodernet.utils.global_config import Config, Backends
from spodernet.utils.logger import Logger, LogLevel
from spodernet.preprocessing.batching import StreamBatcher
from spodernet.preprocessing.pipeline import Pipeline
from spodernet.preprocessing.processors import TargetIdx2MultiTarget
from spodernet.hooks import LossHook, ETAHook
from spodernet.utils.util import Timer
from spodernet.preprocessing.processors import TargetIdx2MultiTarget
np.set_printoptions(precision=3)

cudnn.benchmark = True

# parse console parameters and set global variables
Config.backend = Backends.TORCH
Config.parse_argv(sys.argv)

Config.cuda = True
Config.embedding_dim = 100
# Logger.GLOBAL_LOG_LEVEL = LogLevel.DEBUG


# model_name = 'DistMult_{0}_{1}'.format(Config.input_dropout, Config.dropout)
model_name = '{2}_{0}_{1}'.format(Config.input_dropout, Config.dropout, Config.model_name)
epochs = 1000
load = False
if Config.dataset is None:
    Config.dataset = 'FB15k-237'
model_path = 'saved_models/{0}_{1}.model'.format(Config.dataset, model_name)


# Preprocess knowledge graph using spodernet.
def preprocess(dataset_name, delete_data=False):
    full_path = 'data/{0}/e1rel_to_e2_full.json'.format(dataset_name)
    train_path = 'data/{0}/e1rel_to_e2_train.json'.format(dataset_name)
    dev_ranking_path = 'data/{0}/e1rel_to_e2_ranking_dev.json'.format(dataset_name)
    test_ranking_path = 'data/{0}/e1rel_to_e2_ranking_test.json'.format(dataset_name)
    entity_path = 'data/{0}/{1}.content'.format(dataset_name, dataset_name)
    rel_path = 'data/{0}/{1}.rel'.format(dataset_name, dataset_name)

    entity_feature = np.genfromtxt(entity_path, dtype=np.dtype(str))
    entity_name = entity_feature[:, 0]
    entity_feature = np.array(entity_feature[:, 1:], dtype=np.float32)
    rel_feature = np.genfromtxt(rel_path, dtype=np.dtype(str))
    rel_name = rel_feature[:, 0]
    rel_feature = np.array(rel_feature[:, 2:], dtype=np.float32)

    keys2keys = {'e1': 'e1', 'rel': 'rel', 'rel_eval': 'rel', 'e2': 'e1', 'e2_multi1': 'e1', 'e2_multi2': 'e1'}
    input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
    d = DatasetStreamer(input_keys)
    d.add_stream_processor(JsonLoaderProcessors())
    d.add_stream_processor(DictKey2ListMapper(input_keys))

    # process full vocabulary and save it to disk
    d.set_path(full_path)
    p = Pipeline(Config.dataset, delete_data, keys=input_keys, skip_transformation=True)
    p.add_sent_processor(ToLower())
    p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')), keys=['e2_multi1', 'e2_multi2'])
    p.add_token_processor(AddToVocab())
    p.execute(d)
    p.save_vocabs()

    # process train, dev and test sets and save them to hdf5
    p.skip_transformation = False
    for path, name in zip([train_path, dev_ranking_path, test_ranking_path], ['train', 'dev_ranking', 'test_ranking']):
        d.set_path(path)
        p.clear_processors()
        p.add_sent_processor(ToLower())
        p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')), keys=['e2_multi1', 'e2_multi2'])
        p.add_post_processor(ConvertTokenToIdx(keys2keys=keys2keys), keys=['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2'])
        p.add_post_processor(StreamToHDF5(name, samples_per_file=1000, keys=input_keys))
        p.execute(d)

    # sort entity_feature and rel_feature by p.state['vocab']['e1'].idx2token and p.state['vocab']['rel'].idx2token

    e_idx_token = p.state['vocab']['e1'].idx2token
    e_token = np.array([e_idx_token[idx] for idx in range(2, len(e_idx_token))])
    emb_e_arg = [np.argwhere(entity_name==e)[0][0] for e in e_token]
    emb_e = entity_feature[np.array(emb_e_arg), :]

    rel_idx_token = p.state['vocab']['rel'].idx2token
    rel_token = np.array([rel_idx_token[idx] for idx in range(2, len(rel_idx_token), 2)])
    emb_rel_arg = [np.argwhere(rel_name==rel)[0][0] for rel in rel_token]
    emb_rel = rel_feature[np.array(emb_rel_arg), :]
    emb_rel_rev = -emb_rel
    emb_rel_concat = np.zeros((emb_rel.shape[0] * 2, emb_rel.shape[1]))
    for idx in range(emb_rel.shape[0]):
        emb_rel_concat[idx * 2] = emb_rel[idx]
        emb_rel_concat[idx * 2 + 1] = emb_rel_rev[idx]

    emb_e_fist2 = np.random.normal(size=(2, emb_e.shape[1]))
    emb_e = np.concatenate((emb_e_fist2, emb_e), axis=0)
    emb_rel_fist2 = np.random.normal(size=(2, emb_rel_concat.shape[1]))
    emb_rel = np.concatenate((emb_rel_fist2, emb_rel_concat), axis=0)

    return emb_e, emb_rel


def main():
    emb_e, emb_rel = preprocess(Config.dataset, delete_data=True)
    input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
    p = Pipeline(Config.dataset, keys=input_keys)
    p.load_vocabs()
    vocab = p.state['vocab']

    num_entities = vocab['e1'].num_token

    train_batcher = StreamBatcher(Config.dataset, 'train', Config.batch_size, randomize=True, keys=input_keys)
    dev_rank_batcher = StreamBatcher(Config.dataset, 'dev_ranking', Config.batch_size, randomize=False, loader_threads=4, keys=input_keys)
    test_rank_batcher = StreamBatcher(Config.dataset, 'test_ranking', Config.batch_size, randomize=False, loader_threads=4, keys=input_keys)

    if Config.model_name is None:
        model = ConvE(vocab['e1'].num_token, vocab['rel'].num_token, emb_e)  # 实体数、关系数
    elif Config.model_name == 'ConvE':
        model = ConvE(vocab['e1'].num_token, vocab['rel'].num_token, emb_e)
    elif Config.model_name == 'DistMult':
        model = DistMult(vocab['e1'].num_token, vocab['rel'].num_token, emb_e)
    elif Config.model_name == 'ComplEx':
        model = Complex(vocab['e1'].num_token, vocab['rel'].num_token, emb_e)
    else:
        # log.info('Unknown model: {0}', Config.model_name)
        raise Exception("Unknown model!")

    train_batcher.at_batch_prepared_observers.insert(1, TargetIdx2MultiTarget(num_entities, 'e2_multi1', 'e2_multi1_binary'))

    eta = ETAHook('train', print_every_x_batches=100)
    train_batcher.subscribe_to_events(eta)
    train_batcher.subscribe_to_start_of_epoch_event(eta)
    train_batcher.subscribe_to_events(LossHook('train', print_every_x_batches=100))

    if Config.cuda:
        model.cuda()
    if load:
        model_params = torch.load(model_path)
        print(model)
        total_param_size = []
        params = [(key, value.size(), value.numel()) for key, value in model_params.items()]
        for key, size, count in params:
            total_param_size.append(count)
            print(key, size, count)
        print(np.array(total_param_size).sum())
        model.load_state_dict(model_params)
        model.eval()
        ranking_and_hits(model, test_rank_batcher, vocab, 'test_evaluation')
        ranking_and_hits(model, dev_rank_batcher, vocab, 'dev_evaluation')
    else:
        model.init(emb_e, emb_rel)

    total_param_size = []

    params = {name: value.numel() for name, value in model.named_parameters()}
    print(params)
    # print(np.sum(params))
    opt = torch.optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.L2)
    for epoch in range(epochs):
        model.train()
        for i, str2var in enumerate(train_batcher):
            opt.zero_grad()
            e1 = str2var['e1']  # batch_size * 1
            rel = str2var['rel']  # batch_size * 1
            e2_multi = str2var['e2_multi1_binary'].float()  # batch_size * num_entities
            # label smoothing
            e2_multi = ((1.0-Config.label_smoothing_epsilon)*e2_multi) + (1.0/e2_multi.size(1))

            pred = model.forward(e1, rel)
            loss = model.loss(pred, e2_multi)
            loss.backward()
            opt.step()

            train_batcher.state.loss = loss.cpu()

        print('saving to {0}'.format(model_path))
        torch.save(model.state_dict(), model_path)

        model.eval()
        with torch.no_grad():
            ranking_and_hits(model, dev_rank_batcher, vocab, 'dev_evaluation')
            if epoch % 3 == 0:
                if epoch > 0:
                    ranking_and_hits(model, test_rank_batcher, vocab, 'test_evaluation')


if __name__ == '__main__':
    time1 = time.time()
    main()
    print("total time: {:.1f}s".format(time.time() - time1))
