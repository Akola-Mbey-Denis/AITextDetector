import pandas as pd
import numpy as np
import transformers
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Subset
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm
from models.bert import AITextDetector
from models.roberta_model import RobertaAITextDetector
import argparse
import yaml
from utils.dataloader import AITextDataset
from transformers import AdamW


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/train.cfg',
                        help='config file. see readme')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--data', type=str, default='train',
                        help='Specify the dataset type : train/test')
    parser.add_argument('--max_length', type=int, default=300,
                        help='Maximum length')
    parser.add_argument('--arch', type=str, default='bert',
                        help='Specify the arch type : bert/roberta')
    
    parser.add_argument('--valid_split', type=float, default=0.25,
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
torch.manual_seed(args['SEED'])
np.random.seed(0)
torch.backends.cudnn.deterministic = args['DETERMINISTIC']
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if arch =='bert':
    model = AITextDetector(dropout = args['DROPOUT'])
else:
    model = RobertaAITextDetector(dropout =args['DROPOUT'])
# Fine tune only fc layer
for param in model.bert.parameters():
    param.requires_grad = False
# move model to the right device
model.to(device)

tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

dataset = AITextDataset(tokenizer =tokenizer, max_length = max_length, data_type = dataset_type,file_path =args['PATH'])

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


def validation_loop(dataloader,model):
    model.eval()
    loop = tqdm(enumerate(dataloader),leave=False,total=len(dataloader))
    num_correct = 0
    target_count = 0
    with torch.no_grad():
        for batch, dl in loop:
            ids=dl['ids'].to(device)
            token_type_ids=dl['token_type_ids'].to(device)
            mask= dl['mask'].to(device)
            label=dl['target'].to(device)
            label = label.unsqueeze(1).to(device)
            if arch =='bert':
                output = model(ids,mask,token_type_ids)
            else:
                 output=model(ids,mask)

            label = label.type_as(output)                
            output =output.cpu().detach().numpy()
            pred = np.where(output >= 0, 1, 0)
            target_count += label.size(0)
            # compute accuracy per batch
            num_correct+= sum(1 for a, b in zip(pred, label) if a[0] == b[0])
       
    print(f'Validation accuracy :  {round(float(100 * num_correct / target_count),3)}')
    return float(100 * num_correct / target_count)
    

def training_loop(epochs,dataloader,val_dataloader,model,loss_fn,optimizer):
    model.train()
    for  epoch in range(epochs):       
        loop=tqdm(enumerate(dataloader),leave=False,total=len(dataloader))
        total_loss  = 0
        num_correct = 0
        target_count = 0
        for batch, dl in loop:
            ids=dl['ids'].to(device)
            token_type_ids=dl['token_type_ids'].to(device)
            mask= dl['mask'].to(device)
            label=dl['target'].to(device)
            label = label.unsqueeze(1).to(device)

            optimizer.zero_grad()
            
           if arch =='bert':
                output = model(ids,mask,token_type_ids)
            else:
                output=model(ids,mask)
            label = label.type_as(output)

            loss = loss_fn(output,label)
            # back propagate
            loss.backward()            
            optimizer.step()
            output = output.cpu().detach().numpy()
            pred = np.where(output >= 0, 1, 0)

            target_count += label.size(0)
            
            # compute accuracy per batch
            num_correct+= sum(1 for a, b in zip(pred, label) if a[0] == b[0]) 
              
          
            total_loss+= loss.item() 
            # print(total_loss)
       
        print(f'Training loss :{round(float(total_loss/len(dataloader)),3)}   with accuracy {round(float(100 * num_correct /target_count),3)}')
        if epoch%5:
           val_acc = validation_loop(val_dataloader,model)   
           # Show progress while training
           loop.set_description(f'Epochs={epoch}/{epochs}')
           #saved model 
           torch.save(model.state_dict(), 'new_arch/ai-text-classifier-'+str(val_acc)+'.pth')
           

    return model

# construct an optimizer
optimizer= optimizer = AdamW(model.parameters(),lr =1e-5)   
loss_fn = nn.BCEWithLogitsLoss()
model = training_loop(epochs, train_dataloader,val_dataloader, model, loss_fn, optimizer)
torch.save(model.state_dict(), 'new_arch/ai-text'+arch+'-classifier-max-length-'+str(max_length)+'.pth')



