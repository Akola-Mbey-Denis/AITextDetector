

from transformers import AutoTokenizer, XLMRobertaModel
import torch.nn as nn
import torch
from transformers import XLMRobertaModel
class XLMRobertaAITextDetector(nn.Module):
    def __init__(self, dropout=0.3):
        '''
           GPT2 model with new fully connected layer for text classification
        '''
        super(XLMRobertaAITextDetector, self).__init__()    
        self.dropout_rate = dropout 
        self.base =  XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.fc = nn.Sequential(
            nn.Dropout(p = self.dropout_rate),
            nn.Linear(768,768),
            nn.Tanh(),
            nn.Dropout(p = self.dropout_rate),
            nn.Linear(768,1)
        )
        
        self._init_weights()       
    
        self._init_weights()       
        
    def forward(self, input_ids, attention_mask):
        _, x2 = self.base(input_ids=input_ids, attention_mask=attention_mask,return_dict=False)
        x = self.fc(x2)
        return x
        
    def _init_weights(self):
        """ Initialize the weights """
        for  m in self.fc:
            if isinstance(m, (nn.Linear,nn.LayerNorm)):
                m.weight.data.normal_(mean=0.0, std=0.02)
        
            if isinstance(m, (nn.Linear,nn.LayerNorm)) and m.bias is not None:
                m.bias.data.zero_()
    