from src.process import TextDataset
import os
def get_expert_split(lang_type, tokenizer, args):
    train_dataset = get_datasets_dic(tokenizer, lang_type, args)
    return train_dataset

def get_datasets_dic(tokenizer, lang_type, args):
    # dataset_dic = {}
    # python 251820 java 164923 go 167288 ruby 24927 javascript 58025 php 241241
    train_dataset = TextDataset(tokenizer, args, os.path.join(args.root_data_file, lang_type, args.train_data_file))
    # dataset_dic[lang_type] = train_dataset
    return train_dataset