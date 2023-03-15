import random
import pandas as pd
import numpy as np
import transformers
import torch
import time
import torch.nn as nn
from torch.utils.data import DataLoader,Subset
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm
from models.bert import BertAITextDetector
from models.roberta_model import RobertaAITextDetector
from models.gpt import GPTAITextDetector
import argparse
import yaml
from utils.dataloader import AITextDataset
from transformers import AdamW,get_linear_schedule_with_warmup


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/train.cfg',
                        help='config file. see readme')
    parser.add_argument('--epochs', type=int, default = 50,
                        help='Number of training epochs')
    parser.add_argument('--data', type=str, default='train',
                        help='Specify the dataset type : train/test')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum length')
    parser.add_argument('--arch', type=str, default='bert',
                        help='Specify the arch type : bert/roberta/gpt/xlm')
    
    parser.add_argument('--valid_split', type=float, default=0.20,
                        help='Specify the dataset type : train/test')

    return parser.parse_args()

args = parse_args()
dataset_type = args.data
epochs = args.epochs 
max_length = args.max_length
valid_split = args.valid_split
arch = args.arch

with open(args.cfg, 'r') as f:
        args = yaml.load(f, Loader=yaml.Loader)


# Enforce reproducibility
random.seed(args['SEED'])
torch.manual_seed(args['SEED'])
np.random.seed(args['SEED'])
torch.cuda.manual_seed_all(args['SEED'])
torch.backends.cudnn.deterministic = args['DETERMINISTIC']

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if arch =='bert':
    model = BertAITextDetector(dropout = args['DROPOUT'])
elif arch =='roberta':
    model = RobertaAITextDetector(dropout =args['DROPOUT'])
    print(model)
elif arch == 'gpt':
    model = GPTAITextDetector(dropout =args['DROPOUT'])


# Fine tune only fc layer
if arch!='gpt':
    for param in model.bert.parameters():
        param.requires_grad = False
else:
    for param in model.base.parameters():
        param.requires_grad = False     


# move model to the right device
model.to(device)
if arch =='bert':
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
elif arch =='roberta':
    tokenizer = transformers.RobertaTokenizer.from_pretrained("roberta-base")
elif arch =='gpt':
    # Get model's tokenizer.
    tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"cls_token": "[CLS]"})
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.base.resize_token_embeddings(len(tokenizer))   
elif arch == 'xlm':
    tokenizer = transformers.AutoTokenizer.from_pretrained("xlm-roberta-base")

dataset = AITextDataset(tokenizer =tokenizer, max_length = max_length, data_type = dataset_type,model_type =arch,file_path =args['PATH'])

dataset_size = len(dataset)
validation_split = valid_split
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
train_split = Subset(dataset, train_indices)

val_split = Subset(dataset, val_indices)
train_dataloader = torch.utils.data.DataLoader(train_split, batch_size=args['BATCH_SIZE'], shuffle=True, num_workers=4,collate_fn = dataset.collate_fn_train)
val_dataloader = torch.utils.data.DataLoader(val_split, batch_size=args['BATCH_SIZE'], shuffle=False, num_workers=4,collate_fn= dataset.collate_fn_train)


def validation_loop(dataloader,model,loss_fn):
    model.eval()
    loop = tqdm(enumerate(dataloader),leave=False,total=len(dataloader))
    num_correct = 0
    target_count = 0
    total_loss  = 0
    with torch.no_grad():
        for batch, dl in loop:
            ids=dl['ids'].to(device)
            if arch =='bert':
                token_type_ids=dl['token_type_ids'].to(device)
            mask= dl['mask'].to(device)
            label=dl['target'].to(device)
            label = label.unsqueeze(1).to(device)
            if arch =='bert':
                output = model(ids,mask,token_type_ids)
            else:
                output = model(ids,mask)

            label = label.type_as(output) 
            loss = loss_fn(output,label)               
            output =output.cpu().detach().numpy()
            pred = np.where(output >= 0, 1, 0)
            target_count += label.size(0)
            

            # compute accuracy per batch
            num_correct+= sum(1 for a, b in zip(pred, label) if a[0] == b[0])
            total_loss+= loss.item() 
       
    print(f'Validation loss :{round(float(total_loss/len(dataloader)),3)}   with accuracy {round(float(100 * num_correct /target_count),3)}%')
    return float(100 * num_correct / target_count)
    

def training_loop(epochs,dataloader,val_dataloader,model,loss_fn,optimizer,scheduler):
    model.train()
    for  epoch in range(1,epochs+1):       
        loop=tqdm(enumerate(dataloader),leave=False,total=len(dataloader))
        total_loss  = 0
        num_correct = 0
        target_count = 0
        for batch, dl in loop:
            ids=dl['ids'].to(device)
            if arch =='bert':
                token_type_ids = dl['token_type_ids'].to(device)
            mask= dl['mask'].to(device)
            label=dl['target'].to(device)
            label = label.unsqueeze(1).to(device)

            optimizer.zero_grad()
            
            if arch =='bert':
                output = model(ids,mask,token_type_ids)
            else:
                output = model(ids,mask)
            label = label.type_as(output)

            loss = loss_fn(output,label)

            # back propagate
            loss.backward()  

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)          
            optimizer.step()
            
            output = output.cpu().detach().numpy()
            pred = np.where(output >= 0, 1, 0)

            target_count += label.size(0)
            
            # compute accuracy per batch
            num_correct+= sum(1 for a, b in zip(pred, label) if a[0] == b[0]) 
            total_loss+= loss.item() 
        scheduler.step()
        print(f'Training loss :{round(float(total_loss/len(dataloader)),3)}   with accuracy {round(float(100 * num_correct /target_count),3)}%')
        if epoch%5 == 0:
           val_acc = validation_loop(val_dataloader,model,loss_fn)   
           # Show progress while training
           loop.set_description(f'Epochs={epoch}/{epochs}')
           # saved model 
           torch.save(model.state_dict(), 'fc_bert/ai-text-classifier-'+str(val_acc)+'.pth')
            

    return model

# construct an optimizer
# inspired by https://arxiv.org/pdf/1810.04805.pdf
optimizer = AdamW(model.parameters(), lr=1e-4,eps = 1e-08)

# Set up the learning rate scheduler
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
loss_fn = nn.BCEWithLogitsLoss()

training_start_time = time.time()
model = training_loop(epochs, train_dataloader,val_dataloader, model, loss_fn, optimizer,scheduler)
print('Training finished, took {:.2f}s'.format(time.time() - training_start_time))
torch.save(model.state_dict(), 'fc_bert/ai-text-'+arch+str(epochs)+'-classifier-max-length-'+str(max_length)+'.pth')



