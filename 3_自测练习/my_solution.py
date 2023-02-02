from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import json
import sys
import datetime
import time
import numpy as np
import torch
import torch.nn.functional as F
import pickle
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm, trange
from tensorboardX import SummaryWriter

sys.path.append('/workspace/8.2.1-1/1_算法示例')
from utils import NerProcessor, convert_examples_to_features, get_Dataset
from models import BERT_BiLSTM_CRF
import conlleval

from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer)
from pytorch_transformers import AdamW, WarmupLinearSchedule

logger = logging.getLogger(__name__)

# set the random seed for repeat
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def get_entities(pred_ner, text):
    token_types = [[] for _ in range(len(pred_ner))]
    entities = [[] for _ in range(len(pred_ner))]
    for i in range(len(pred_ner)):
        token_type = []
        entity = []
        j = 0
        word_begin = False
        while j < len(pred_ner[i]):
            if pred_ner[i][j][0] == 'B':
                if word_begin:
                    token_type = []  # 防止多个B出现在一起
                    entity = []
                token_type.append(pred_ner[i][j])
                entity.append(text[i][j])
                word_begin = True
            elif pred_ner[i][j][0] == 'I':
                if word_begin:
                    token_type.append(pred_ner[i][j])
                    entity.append(text[i][j])
            else:
                if word_begin:
                    token_types[i].append(''.join(token_type))
                    token_type = []
                    entities[i].append(''.join(entity))
                    entity = []
                word_begin = False
            j += 1
    return token_types, entities


def test():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file", default='/workspace/8.2.1-1/1_算法示例/data/train.txt', type=str)
    parser.add_argument("--eval_file", default='/workspace/8.2.1-1/1_算法示例/data/dev.txt', type=str)
    parser.add_argument("--test_file", default='/workspace/8.2.1-1/1_算法示例/data/test.txt', type=str)
    parser.add_argument("--model_name_or_path", default='bert-base-chinese', type=str)
    parser.add_argument("--output_dir", default='/workspace/8.2.1-1/1_算法示例/model', type=str)

    ## other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    
    parser.add_argument("--do_train", default=False, type=boolean_string)
    parser.add_argument("--do_eval", default=False, type=boolean_string)
    parser.add_argument("--do_test", default=True, type=boolean_string)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--num_train_epochs", default=10, type=float)
    parser.add_argument("--warmup_proprotion", default=0.1, type=float)
    parser.add_argument("--use_weight", default=1, type=int)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--fp16", default=False)
    parser.add_argument("--loss_scale", type=float, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--logging_steps", default=500, type=int)
    parser.add_argument("--clean", default=False, type=boolean_string, help="clean the output dir")

    parser.add_argument("--need_birnn", default=False, type=boolean_string)
    parser.add_argument("--rnn_dim", default=128, type=int)

    args = parser.parse_args()

    # device = torch.device("cuda")
    device = torch.device("cpu")
    args.device = device
    n_gpu = torch.cuda.device_count()

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)

    logger.info(f"device: {device} n_gpu: {n_gpu}")

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))


    if args.clean and args.do_train:
        # logger.info("清理")
        if os.path.exists(args.output_dir):
            def del_file(path):
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    print(c_path)
                    if os.path.isdir(c_path):
                        del_file(c_path)
                        os.rmdir(c_path)
                    else:
                        os.remove(c_path)
            try:
                del_file(args.output_dir)
            except Exception as e:
                print(e)
                print('pleace remove the files of output dir and data.conf')
                exit(-1)
    
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(os.path.join(args.output_dir, "eval")):
        os.makedirs(os.path.join(args.output_dir, "eval"))
    
    writer = SummaryWriter(logdir=os.path.join(args.output_dir, "eval"), comment="Linear")

    processor = NerProcessor()
    label_list = processor.get_labels(args)
    num_labels = len(label_list)
    args.label_list = label_list

    if os.path.exists(os.path.join(args.output_dir, "label2id.pkl")):
        with open(os.path.join(args.output_dir, "label2id.pkl"), "rb") as f:
            label2id = pickle.load(f)
    else:
        label2id = {l:i for i,l in enumerate(label_list)}
        with open(os.path.join(args.output_dir, "label2id.pkl"), "wb") as f:
            pickle.dump(label2id, f)      
    
    id2label = {value:key for key,value in label2id.items()} 

    label_map = {i : label for i, label in enumerate(label_list)}

    tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    # args = torch.load(os.path.join(args.output_dir, 'training_args.bin'))
    args = torch.load(os.path.join(r'/workspace/8.2.1-1/1_算法示例/model', 'training_args.bin'))
    model = BERT_BiLSTM_CRF.from_pretrained(r'/workspace/8.2.1-1/1_算法示例/model', need_birnn=args.need_birnn, rnn_dim=args.rnn_dim)
    model.to(device)

    # 补全1
    test_examples, test_features, test_data = get_Dataset(args, processor, tokenizer, mode="test")
    # raise NotImplementedError('补全输入方式')


    logger.info("***** Running test *****")
    logger.info(f" Num examples = {len(test_examples)}")
    logger.info(f" Batch size = {args.eval_batch_size}")

    all_ori_tokens = [f.ori_tokens for f in test_features]
    all_ori_labels = [e.label.split(" ") for e in test_examples]
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
    model.eval()

    pred_labels = []
    
    for b_i, (input_ids, input_mask, segment_ids, label_ids) in enumerate(tqdm(test_dataloader, desc="Predicting")):
        
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model.predict(input_ids, segment_ids, input_mask)
        # logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        # logits = logits.detach().cpu().numpy()

        for l in logits:

            pred_label = []
            for idx in l:
                pred_label.append(id2label[idx])
            pred_labels.append(pred_label)

    assert len(pred_labels) == len(all_ori_tokens) == len(all_ori_labels)
    print(len(pred_labels))
    with open('/workspace/8.2.1-1/1_算法示例/model'+'/'+"token_labels_.txt", "w", encoding="utf-8") as f:
        for ori_tokens, ori_labels,prel in zip(all_ori_tokens, all_ori_labels, pred_labels):
            for ot,ol,pl in zip(ori_tokens, ori_labels, prel):
                if ot in ["[CLS]", "[SEP]"]:
                    continue
                else:
                    f.write(f"{ot} {pl}\n")
            f.write("\n")
    # 补全2
    token_types, entities = get_entities(pred_labels, all_ori_tokens)
    # raise NotImplementedError('补全输入方式')
    
    return entities

if __name__ == "__main__":
    res = test()
    print('识别出来的实体如下:')
    print(res)