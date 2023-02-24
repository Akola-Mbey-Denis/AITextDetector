import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
class AITextDataset(Dataset):
    def __init__(self, tokenizer,max_length, data_type ='train',file_path ="./dataset/train_set.json"):
        super(AITextDataset, self).__init__()
        self.path = file_path
        self.data = pd.read_json(self.path)
        self.tokenizer = tokenizer
        self.target = self.data.iloc[:,1]
        self.max_length = max_length
        
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

        return {
            'text_id': self.data.iloc[index,0],
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'target': torch.tensor(self.data.iloc[index, 2], dtype=torch.long)
            }
   