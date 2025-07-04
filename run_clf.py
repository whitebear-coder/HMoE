# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
import json
from src.pre_dataset import get_expert_split
from src.process import TextDataset, InputFeatures

from tqdm import tqdm, trange
import multiprocessing
from model import CLFModel
# from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup, RobertaModel, RobertaConfig, RobertaTokenizer)
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup)
from src.model.roberta_model import RobertaModel
from src.model.configuration_roberta import RobertaConfig
from src.model.tokenization_roberta import RobertaTokenizer

from src.process import loss_fn

from torch import nn

logger = logging.getLogger(__name__)


class InputFeaturesCLF(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label=label


def convert_examples_to_features_CLF(js,tokenizer,args):
    #source
    code=' '.join(js['code'].split())
    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeaturesCLF(source_tokens,source_ids,js['label'])

class CodeData(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        dic_lang = {"java":0, "javascript":1, "php":2, "python":3, "ruby":4, "go":5}
        self.examples = []
        for lang in dic_lang:
            sum_lang = 0
            file_path_lang = os.path.join(args.root_data_file, lang, file_path)
            with open(file_path_lang) as f:
                for line in f:
                    js=json.loads(line.strip())
                    code=' '.join(js['code'].split())
                    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
                    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
                    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
                    padding_length = args.block_size - len(source_ids)
                    source_ids+=[tokenizer.pad_token_id]*padding_length
                    inputFeatures = InputFeaturesCLF(source_tokens,source_ids,dic_lang[lang])
                    self.examples.append(inputFeatures)
                    sum_lang += 1
                    if sum_lang >= 10000:
                        break
            # print("lang_type:{}, sum:{}, file_path:{}".format(lang, sum_lang, file_path_lang))
                
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train(args, train_dataset, model, tokenizer):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=64)



    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    max_steps = len(train_dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps*0.1,
                                                num_training_steps=max_steps)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader.dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", max_steps)
    best_acc=0.0
    model.zero_grad()
    model.train()
    criterion = nn.CrossEntropyLoss()

    for idx in range(10):
    # for idx in range(args.num_train_epochs):
        # bar = tqdm(train_dataloader,total=len(train_dataloader))
        losses=[]
        output_hd = []
        for step, batch in enumerate(train_dataloader):
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            labels = torch.nn.functional.one_hot(labels, args.num_labels)
            y_out, output_hs = model(input_ids=inputs)
            # print(labels)
            labels = labels.float()
            ##### loss部分 #####
            loss = criterion(y_out, labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            losses.append(loss.item())
            # bar.set_description("epoch {} loss {}".format(idx,round(np.mean(losses),3)))
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
        
        with open('layer_outputs_{}.pkl'.format(idx), 'wb') as f:
            pickle.dump(output_hd, f)
        
        
        results = evaluate(args, model, tokenizer)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))

        # Save model checkpoint
        if results['eval_acc']>best_acc:
            best_acc=results['eval_acc']
            logger.info("  "+"*"*20)
            logger.info("  Best acc:%s",round(best_acc,4))
            logger.info("  "+"*"*20)

            checkpoint_prefix = 'checkpoint-best-acc'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model,'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)
       
def evaluate(args, model, tokenizer):
    eval_output_dir = args.output_dir

    eval_dataset = CodeData(tokenizer, args, args.eval_data_file)

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    
    logits=[]
    labels=[]
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        label=batch[1].to(args.device)
        label = torch.nn.functional.one_hot(label, args.num_labels)

        with torch.no_grad():
            logit, b = model(inputs)
            # eval_loss += lm_loss.mean().item()
            eval_loss += -torch.sum(logit*label)
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits=np.concatenate(logits,0)
    labels=np.concatenate(labels,0)
    preds=logits.argmax(-1)
    labels=labels.argmax(-1)
    eval_acc=np.mean(labels==preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    result = {
        "eval_loss": float(perplexity),
        "eval_acc":round(eval_acc,4),
    }
    return result

def test(args, model, tokenizer):
    # Note that DistributedSampler samples randomly
    eval_dataset = CodeData(tokenizer, args, args.test_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]
    labels=[]
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        label=batch[1].to(args.device)
        label = torch.nn.functional.one_hot(label, args.num_labels)
        with torch.no_grad():
            logit,_ = model(inputs)
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits=np.concatenate(logits,0)
    labels=np.concatenate(labels,0)
    preds=logits.argmax(-1)
    labels=labels.argmax(-1)
    # print(preds.shape, labels.shape)
    test_acc = np.mean(labels==preds)

    result = {
        "test_acc":round(test_acc,4),
    }
    return result

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_data_file", default=None, type=str, required=True,
                    help="The input root data file (a text file).")
    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--num_train_epochs', type=int, default=42,
                        help="num_train_epochs")
    parser.add_argument('--num_labels', type=int, default=None,
                        help = 'num_labels')
    parser.add_argument("--datasets", type=str, default=[], nargs="+")
    parser.add_argument('--root_dir', type=str, help="dataset dir")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")
    

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args.seed)
    config = RobertaConfig.from_pretrained("/home/linzexu/huggingface/cocosoda")
    tokenizer = RobertaTokenizer.from_pretrained("/home/linzexu/huggingface/cocosoda")
    model = RobertaModel.from_pretrained("/home/linzexu/huggingface/cocosoda", output_hidden_states=True)
    model = CLFModel(model)
    
    # multi-gpu training (should be after apex fp16 initialization)
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        model = model.cuda()
    
    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        # train_data_file = os.path.join("/home/linzexu/meta_code/Meta-DMoE-main/CSN")
        train_dataset = CodeData(tokenizer, args, args.train_data_file)
        train(args, train_dataset, model, tokenizer)

    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        # model.load_state_dict(torch.load(output_dir))
        checkpoint = torch.load(output_dir)
        for key in list(checkpoint.keys()):
            # if 'module.' in key:
            new_key = 'module.' + key
            checkpoint[new_key] = checkpoint[key]
            # checkpoint[key.replace('module.', '')] = checkpoint[key]
            del checkpoint[key]
        model.load_state_dict(checkpoint)      
        model.to(args.device)
        result = test(args, model, tokenizer)
        logger.info("***** Final Test results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))
   
    

if __name__ == "__main__":
    main()
