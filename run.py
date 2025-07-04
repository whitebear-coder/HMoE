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
import sys 
import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
from model import Model, CLFModel, MLP
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
# from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
#                               RobertaConfig, RobertaModel, RobertaTokenizer)

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup)

from src.model.roberta_model import RobertaModel
from src.model.configuration_roberta import RobertaConfig
from src.model.tokenization_roberta import RobertaTokenizer
from cores_model import GPO_Model

logger = logging.getLogger(__name__)

def polyloss(view1, view2, margin):
    
    sim_mat = sim_matrix(view1,view2)
    epsilon = 1e-5
    size=sim_mat.size(0)
    hh=sim_mat.t()
    label=torch.Tensor([i for i in range(size)])
  
    loss = list()
    for i in range(size):
        pos_pair_ = sim_mat[i][i]
        pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
        neg_pair_ = sim_mat[i][label!=label[i]]

        neg_pair = neg_pair_[neg_pair_ + margin > min(pos_pair_)]

        pos_pair=pos_pair_
        if len(neg_pair) < 1 or len(pos_pair) < 1:
            continue

        pos_loss =torch.clamp(0.2*torch.pow(pos_pair,2)-0.7*pos_pair+0.5, min=0)
        neg_pair=max(neg_pair)
        neg_loss = torch.clamp(0.9*torch.pow(neg_pair,2)-0.4*neg_pair+0.03,min=0)

        loss.append(pos_loss + neg_loss)
    for i in range(size):
        pos_pair_ = hh[i][i]
        pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
        neg_pair_ = hh[i][label!=label[i]]

        neg_pair = neg_pair_[neg_pair_ + margin > min(pos_pair_)]

        pos_pair=pos_pair_
        if len(neg_pair) < 1 or len(pos_pair) < 1:
            continue
        pos_loss =torch.clamp(0.2*torch.pow(pos_pair,2)-0.7*pos_pair+0.5,min=0)

        neg_pair=max(neg_pair)
        neg_loss = torch.clamp(0.9*torch.pow(neg_pair,2)-0.4*neg_pair+0.03,min=0)
        loss.append(pos_loss + neg_loss)
        
    if len(loss) == 0:
        return torch.zeros([], requires_grad=True)

    loss = sum(loss) / size
    return loss


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt
class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 url,

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url

        
def convert_examples_to_features(js,tokenizer,args):
    """convert examples to token ids"""
    code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
    code_tokens = tokenizer.tokenize(code)[:args.code_length-4]
    code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length
    
    nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length-4]
    nl_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length    
    
    return InputFeatures(code_tokens,code_ids,nl_tokens,nl_ids,js['url'] if "url" in js else js["retrieval_idx"])

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        data = []
        with open(file_path) as f:
            if "jsonl" in file_path:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    if 'function_tokens' in js:
                        js['code_tokens'] = js['function_tokens']
                    data.append(js)
            elif "codebase"in file_path or "code_idx_map" in file_path:
                js = json.load(f)
                for key in js:
                    temp = {}
                    temp['code_tokens'] = key.split()
                    temp["retrieval_idx"] = js[key]
                    temp['doc'] = ""
                    temp['docstring_tokens'] = ""
                    data.append(temp)
            elif "json" in file_path:
                for js in json.load(f):
                    data.append(js) 

        for js in data:
            self.examples.append(convert_examples_to_features(js,tokenizer,args))
                
        if "train" in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))                             
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].nl_ids))
            

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def loss_fn(nl_vec,code_vec):
    poly_loss = polyloss(nl_vec,code_vec,0.15)
    return poly_loss

def train(args, model, ori_model, clf_model, tokenizer, expert_model_list):
    train_data = {}
    for lang_type in ["ruby", "javascript", "go", "java", "php", "python"]:
        try:
            with open(f"model/{lang_type}/pre_embedding_num_{args.datasets_len}_batchSize_{args.train_batch_size}.pkl", "rb") as f:
                train_data[lang_type] = pickle.load(f)
        except FileNotFoundError:
        # 字典存储属于某类 data: class
            train_data[lang_type] = get_expert_split(lang_type, tokenizer, args)
            file_path = f"model/{lang_type}/pre_embedding_num_{args.datasets_len}_batchSize_{args.train_batch_size}.pkl"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(f"model/{lang_type}/pre_embedding_num_{args.datasets_len}_batchSize_{args.train_batch_size}.pkl", "wb") as f:
                pickle.dump(train_data[lang_type], f)
    
    """ Train the model """
    # 上述代码替换
    # train_dataset = TextDataset(tokenizer, args, args.train_data_file)
    #
    for lang_type in ["ruby", "javascript", "go", "java", "php", "python"]:
        # print(lang_type)
        train_sampler = RandomSampler(train_data[lang_type])
        train_dataloader = DataLoader(train_data[lang_type], sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
        
        #get optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader) * args.num_train_epochs)


        # TODO ctr_trainer
        # ctr_trainer.fit(train_dataloader, val_dataloader=None, exp_d=args.exp_d, exp_t=args.exp_t, bal_d=args.bal_d, bal_t=args.bal_t, domain_num=domain_num, task_num=task_num)


        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data[lang_type]))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
        logger.info("  Total train batch size  = %d", args.train_batch_size)
        logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
        
        # model.resize_token_embeddings(len(tokenizer))
        model.zero_grad()
        
        model.train()
        # print(lang_type)
        tr_num,tr_loss,best_mrr = 0,0,0
        for idx in range(args.num_train_epochs): 
            for step,batch in enumerate(train_dataloader):
                #get inputs
                code_inputs = batch[0].to(args.device)    
                nl_inputs = batch[1].to(args.device)
                # TODO 多专家
                '''
                for i in range(6):
                '''
                # for x in range(6):
                code_prefix = []
                with torch.no_grad():
                    outputs, hidden_states = clf_model(input_ids=code_inputs)
                    for i, state in enumerate(hidden_states):
                        state.to(args.device)
                        code_prefix.append(state)

                code_vec = model(code_inputs=code_inputs, code_prefix=code_prefix)
                # code_vec = aggModel(code_vec)

                nl_vec = model(nl_inputs=nl_inputs)
                '''
                code_prefix = []
                with torch.no_grad():
                    outputs, hidden_states = clf_model(input_ids=code_inputs)
                    for i, state in enumerate(hidden_states):
                        state.to(args.device)
                        code_prefix.append(state)  
                
                code_vec, _ = model(code_inputs=code_inputs, code_prefix=code_prefix)
                nl_vec, _ = model(nl_inputs=nl_inputs)
                '''
                #calculate scores and loss
                loss = loss_fn(code_vec,nl_vec)

                #report loss
                tr_loss += loss.item()
                tr_num += 1
                if (step+1)%100 == 0:
                    logger.info("epoch {} step {} loss {}".format(idx,step+1,round(tr_loss/tr_num,5)))
                    tr_loss = 0
                    tr_num = 0
                
                #backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step() 
                
            #evaluate    
            results = evaluate(args, model, tokenizer,lang_type, args.eval_data_file, eval_when_training=True)
            for key, value in results.items():
                logger.info("  %s = %s", key, round(value,4))    
                
            #save best model
            if results['eval_mrr']>best_mrr:
                best_mrr = results['eval_mrr']
                logger.info("  "+"*"*20)  
                logger.info("  Best mrr:%s",round(best_mrr,4))
                logger.info("  "+"*"*20)                          

                checkpoint_prefix = 'checkpoint-best-mrr'
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)                        
                model_to_save = model.module if hasattr(model,'module') else model
                output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                torch.save(model_to_save.state_dict(), output_dir)
                logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer,lang_type, file_sub_name, eval_when_training=False):
    file_name = os.path.join(args.root_data_file, lang_type, file_sub_name)
    query_dataset = TextDataset(tokenizer, args, file_name)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size,num_workers=4)
    file_name_codebase = os.path.join(args.root_data_file, lang_type, args.codebase_file)
    code_dataset = TextDataset(tokenizer, args, file_name_codebase)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size,num_workers=4)    

    
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    # 加model.eval()之后获得结果一样
    print("evaluateCS ok")
    model.eval()
    code_vecs = [] 
    nl_vecs = []
    for batch in query_dataloader:  
        nl_inputs = batch[1].to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs) 
            nl_vecs.append(nl_vec.cpu().numpy()) 

    for batch in code_dataloader:
        code_inputs = batch[0].to(args.device)    
        with torch.no_grad():
            code_vec = model(code_inputs=code_inputs)
            code_vecs.append(code_vec.cpu().numpy())  
    model.train()
    print("恢复train ok")  
    code_vecs = np.concatenate(code_vecs,0)
    nl_vecs = np.concatenate(nl_vecs,0)
    
    scores = np.matmul(nl_vecs,code_vecs.T)
    # logger.info("  scores shape= {}", scores.shape)
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]    
    # logger.info("  sort_ids shape= {}", sort_ids.shape)
    nl_urls = []
    code_urls = []
    for example in query_dataset.examples:
        nl_urls.append(example.url)
        
    for example in code_dataset.examples:
        code_urls.append(example.url)

    ranks = []
    for url, sort_id in zip(nl_urls,sort_ids):
        rank = 0
        find = False
        for idx in sort_id[:1000]:
            if find is False:
                rank += 1
            if code_urls[idx] == url:
                find = True
        if find:
            ranks.append(1/rank)
        else:
            ranks.append(0)

    result = {
        "eval_mrr":float(np.mean(ranks))
    }

    return result
                       
                        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, 
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--root_data_file", default="/home/linzexu/meta_code/Meta-DMoE-main/CSN", type=str,
                    help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_clf", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")  
    parser.add_argument("--do_zero_shot", action='store_true',
                        help="Whether to run eval on the test set.")     
    parser.add_argument("--do_F2_norm", action='store_true',
                        help="Whether to run eval on the test set.")      

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--datasets_len", default="12345678910", type=str,
                    help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    
    #print arguments
    args = parser.parse_args()
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)

    #build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    ori_model = RobertaModel.from_pretrained(args.model_name_or_path, output_hidden_states=True) 
    ori_model1 = RobertaModel.from_pretrained(args.model_name_or_path, output_hidden_states=True)

    ###
    expert_model_list = []
    for i in range(6):
        c1 = RobertaModel.from_pretrained(args.model_name_or_path, output_hidden_states=True) 
        c2 = GPO_Model(c1)
        if args.n_gpu > 1:
            c2 = torch.nn.DataParallel(c2)
        expert_model_list.append(c2)

    ###
    clf_model = CLFModel(ori_model1)
    clf_model.load_state_dict(torch.load("/home/linzexu/Mixup4Code/CodeBERT/codesearch/saved_models/checkpoint-best-acc/model.bin"))
    model = GPO_Model(ori_model)
    logger.info("Training/evaluation parameters %s", args)
    # model_to_load = model.module if hasattr(model, 'module') else model  
    # model_to_load.load_state_dict(torch.load("saved_models/CSN/ruby/checkpoint-best-mrr/model.bin"))  
    model.to(args.device)
    clf_model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        # clf_model = torch.nn.DataParallel(clf_model) 

    if args.do_train:
        train(args, model, ori_model, clf_model, tokenizer, expert_model_list)
        # train(args, model, ori_model, tokenizer)
    results = {}
    '''
    if args.do_eval:
        if args.do_zero_shot is False:
            checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model_to_load = model.module if hasattr(model, 'module') else model  
            model_to_load.load_state_dict(torch.load(output_dir))      
        model.to(args.device)
        result = evaluate(args, model, tokenizer,args.eval_data_file)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],3)))
    '''        
    if args.do_test:
        model_to_load = model.module if hasattr(model, 'module') else model  
        model_to_load.load_state_dict(torch.load("saved_models/CSN/ruby/checkpoint-best-mrr/model.bin"))      
        model.to(args.device)
        for lang_type in ["ruby", "javascript", "go", "java", "php", "python"]:
            # checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
            # output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            result = evaluate(args, model, tokenizer, lang_type, args.test_data_file)
            print(lang_type)
            logger.info("***** Eval results *****")
            
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(round(result[key],3)))


if __name__ == "__main__":
    perception = torch.nn.Linear(6, 1)
    main()