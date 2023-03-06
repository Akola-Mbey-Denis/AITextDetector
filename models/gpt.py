from transformers import GPT2Model
import torch.nn as nn
import torch
class GPTAITextDetector(nn.Module):
    def __init__(self, dropout=0.3):
        '''
           GPT2 model with new fully connected layer for text classification
        '''
        super(GPTAITextDetector, self).__init__()    
        self.dropout_rate = dropout 
        self.base =  GPT2Model.from_pretrained('gpt2')
        self.fc = nn.Sequential(
            nn.Linear(768,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
        self._init_weights()       
        
    def forward(self, input_ids, attention_mask):
        '''
        Inspire by :
                    https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/models/gpt2/modeling_gpt2.py#L1328
        
        '''
        x = self.base(input_ids=input_ids, attention_mask=attention_mask,return_dict=True)
        sequence = x[0]
   
        logits = self.fc(sequence)
        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]

        if self.base.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.ne(input_ids, self.base.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1
        
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        return pooled_logits
        
    def _init_weights(self):
        """ Initialize the weights """
        for  m in self.fc:
            if isinstance(m, (nn.Linear,nn.LayerNorm)):
                m.weight.data.normal_(mean=0.0, std=0.02)
        
            if isinstance(m, (nn.Linear,nn.LayerNorm)) and m.bias is not None:
                m.bias.data.zero_()
    