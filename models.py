import transformers
import torch.nn as nn
class AITextDector(nn.Module):
    def __init__(self,dropout= 0.20, base_model ='bert-base'):
        super(AITextDector, self).__init__()
        self.bert = baseModel(name=base_model).pretrained_model()
        self.dropout =  dropout
        self.fc = nn.Sequential(
          nn.Linear(768,256),
          nn.ReLU(),
          nn.Dropout(p = self.dropout),
          nn.Linear(256,1)
        )
        
    def forward(self,ids,mask,token_type_ids):
        _,o2= self.bert(ids,attention_mask=mask,token_type_ids=token_type_ids, return_dict=False)
        
        out= self.fc(o2)
        
        return out
    

class baseModel:
    def __init__(self, name ="bert-base"):
        self.name = name

    def pretrained_model(self):
        if self.name == 'bert-base':
            return transformers.BertModel.from_pretrained('bert-base-uncased')
        elif self.name =='bert-large':
            return transformers.BertModel.from_pretrained('bert-large-uncased')

