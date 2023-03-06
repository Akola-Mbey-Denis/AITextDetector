import transformers
import torch.nn as nn
class BertAITextDetector(nn.Module):
    def __init__(self,dropout= 0.20, base_model ='bert-base'):
        '''
           Bert model with a new fully connected layer for text classification
        '''
        super(BertAITextDetector, self).__init__()
        self.bert = baseModel(name=base_model).pretrained_model()
        self.dropout =  dropout
        self.fc = nn.Sequential(
            nn.Dropout(p = self.dropout),        
            nn.Linear(768,1)
        )
        self._init_weights()
        
    def forward(self,ids,mask,token_type_ids):
        _,o2= self.bert(ids,attention_mask=mask,token_type_ids=token_type_ids, return_dict=False)
        
        out= self.fc(o2)
        
        return out

    def _init_weights(self):
        """ Initialize the weights """
        for  m in self.fc:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
        
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
    

class baseModel:
    def __init__(self, name ="bert-base"):
        self.name = name

    def pretrained_model(self):
        if self.name == 'bert-base':
            return transformers.BertModel.from_pretrained('bert-base-uncased')
        elif self.name =='bert-large':
            return transformers.BertModel.from_pretrained('bert-large-uncased')

