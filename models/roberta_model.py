import transformers
import torch.nn as nn
class RobertaAITextDetector(nn.Module):
    def __init__(self, dropout=0.3):
        '''
           Roberta model with new fully connected layer for text classification
        '''
        super(RobertaAITextDetector, self).__init__()    
        self.dropout_rate = dropout    
        self.bert = transformers.RobertaModel.from_pretrained('roberta-base')
        self.fc = nn.Sequential(
            nn.Linear(768,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
        self._init_weights()       
        
    def forward(self, input_ids, attention_mask):
        _, x2 = self.bert(input_ids=input_ids, attention_mask=attention_mask,return_dict=False)
        x = self.fc(x2)
        return x
        
    def _init_weights(self):
        """ Initialize the weights """
        for  m in self.fc:
            if isinstance(m, (nn.Linear,nn.LayerNorm)):
                m.weight.data.normal_(mean=0.0, std=0.02)
        
            if isinstance(m, (nn.Linear,nn.LayerNorm)) and m.bias is not None:
                m.bias.data.zero_()
    