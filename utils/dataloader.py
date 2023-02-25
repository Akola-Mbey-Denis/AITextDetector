import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import warnings
warnings.filterwarnings('ignore')
class AITextDataset(Dataset):
    def __init__(self, tokenizer,max_length, data_type ='train',file_path ="./dataset/train_set.json"):
        super(AITextDataset, self).__init__()
        self.path = file_path
        self.data = pd.read_json(self.path)
        self.tokenizer = tokenizer
        self.target = self.data.iloc[:,1]
        self.max_length = max_length
        self.data_type = data_type
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        text = self.data.iloc[index,1]
        
        inputs = self.tokenizer.encode_plus(
            text ,
            None,
            padding = True,
            add_special_tokens = True,
            return_attention_mask = True,
            truncation = True,
            max_length = self.max_length,
        )
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]
        if self.data_type == 'train':
            return {
                'text_id': self.data.iloc[index,0],
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),

                'target': torch.tensor(self.data.iloc[index, 2],dtype=torch.long)
                }
        else:
            return {
                'text_id': self.data.iloc[index,0],
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
                }
   

    @staticmethod
    def collate_fn_train(data):
        ids =[torch.tensor(d['ids']) for d in data]
        masks =[torch.tensor(d['mask']) for d in data]
        token_type_ids = [torch.tensor(d['token_type_ids']) for d in data]
        labels = [d['target'] for d in data]
        text_ids = [d['text_id'] for d in data]
        ids_ = pad_sequence(ids, batch_first=True)
        masks_ = pad_sequence(masks, batch_first=True)
        token_type_ids_= pad_sequence(token_type_ids, batch_first=True)
        labels_ = torch.tensor(labels)
        text_id_ = torch.tensor(text_ids)
        return {
            'text_id': text_id_,
            'ids': ids_,
            'mask': masks_,
            'token_type_ids': token_type_ids_,
            'target': labels_
            }
    @staticmethod
    def collate_fn_test(data):
        ids= [torch.tensor(d['ids']) for d in data]
        masks =[torch.tensor(d['mask']) for d in data]
        token_type_ids = [torch.tensor(d['token_type_ids']) for d in data]
        labels = [d['target'] for d in data]
        ids_ = pad_sequence(ids, batch_first=True)
        masks_ = pad_sequence(masks, batch_first=True)
        token_type_ids_ = pad_sequence(token_type_ids, batch_first=True)
        labels_ = torch.tensor(labels)
        text_ids = [d['text_id'] for d in data]
        text_id_ =  torch.tensor(text_ids)
    
        return {
            'text_id': text_id_,
            'ids': ids_,
            'mask': masks_,
            'token_type_ids': token_type_ids_,
        }