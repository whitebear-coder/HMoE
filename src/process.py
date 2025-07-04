from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
import torch
import numpy as np
import random
import json
import logging
import os
logger = logging.getLogger(__name__)

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def loss_fn(nl_vec,code_vec):

    poly_loss = polyloss(nl_vec,code_vec,0.15)

    return poly_loss

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

def convert_examples_to_features(js,tokenizer,args):
    """convert examples to token ids"""
    # 代码片段预处理
    code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
    code_tokens = tokenizer.tokenize(code)[:args.code_length-4]
    code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    # 将token 转换成 id
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    # 加padding 补齐
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length
    
    # 自然语言预处理
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
                # 对每行进行处理
                for line in f:
                    # 删除空格和换行符和t
                    line = line.strip()
                    # json.load()方法是从json文件读取json，而json.loads()方法是直接读取json，两者都是将字符串json转换为字典
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
            # 将js字典转换
            self.examples.append(convert_examples_to_features(js,tokenizer,args))
            '''
            examples包括以下几个部分
                    self.code_tokens = code_tokens
                    self.code_ids = code_ids
                    self.nl_tokens = nl_tokens
                    self.nl_ids = nl_ids
                    self.url = url
            '''
                
        if "train" in file_path:
            for idx, example in enumerate(self.examples[:1]):
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

def set_seed(seed):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
  
class CustomDataset(Dataset):
    def __init__(self, code, nl):
        self.code = code
        self.nl = nl

    def __len__(self):
        return len(self.code)

    def __getitem__(self, idx):
        return self.code[idx], self.nl[idx]

def save_model(model, name, epoch, test_way='ood'):
    if not os.path.exists("model/code_nl"):
        os.makedirs("model/code_nl")
    path = "model/code_nl/{}_{}_best.bin".format(name, epoch)
    torch.save(model.state_dict(), path)
