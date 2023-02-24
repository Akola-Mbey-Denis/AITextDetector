import pandas as pd
import numpy as np
import transformers
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm
from models import AITextDector
import argparse
import yaml
from utils.dataloader import AITextDataset
from torch.utils.data.sampler import SubsetRandomSampler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/train.cfg',
                        help='config file. see readme')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--data', type=str, default='train',
                        help='Specify the dataset type : train/test')
    parser.add_argument('--max_length', type=int, default=100,
                        help='Maximum length')
    
    parser.add_argument('--valid_split', type=float, default=0.25,
                        help='Specify the dataset type : train/test')

    return parser.parse_args()

args = parse_args()
dataset_type = args.data
epochs = args.epochs 
max_length = args.max_length
valid_split = args.valid_split



with open(args.cfg, 'r') as f:
        args = yaml.load(f, Loader=yaml.Loader)

torch.manual_seed(args['SEED'])
torch.backends.cudnn.deterministic = args['DETERMINISTIC']
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


model = AITextDector(dropout = args['DROPOUT'])
# Fine tune only fc layer
for param in model.bert.parameters():
    param.requires_grad = False
# move model to the right device
model.to(device)

tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
#tokenizer,max_length, data_type ='train',file_path ="./dataset/train_set.json"
dataset = AITextDataset(tokenizer =tokenizer, max_length = max_length, data_type = dataset_type,file_path =args['PATH'])
dataset_size = len(dataset)
validation_split = valid_split
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args['BATCH_SIZE'], shuffle=False, num_workers=4,sampler=train_sampler)
val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args['BATCH_SIZE'], shuffle=False, num_workers=4, sampler=valid_sampler)


def validation_loop(dataloader,model):
    model.eval()
    loop = tqdm(enumerate(dataloader),leave=False,total=len(dataloader))
    for batch, dl in loop:
        ids=dl['ids'].to(device)
        token_type_ids=dl['token_type_ids'].to(device)
        mask= dl['mask'].to(device)
        label=dl['target'].to(device)
        label = label.unsqueeze(1).to(device)
        output=model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids)
        label = label.type_as(output)

            
        output =output.cpu().detach().numpy()
        pred = np.where(output >= 0, 1, 0)

        num_correct = sum(1 for a, b in zip(pred, label) if a[0] == b[0])
        num_samples = pred.shape[0]
        accuracy = num_correct/num_samples
    print(f'Got Validation: {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    

def training_loop(epochs,dataloader,val_dataloader,model,loss_fn,optimizer):
    model.train()
    for  epoch in range(epochs):
    
        
        loop=tqdm(enumerate(dataloader),leave=False,total=len(dataloader))
        for batch, dl in loop:
            ids=dl['ids'].to(device)
            token_type_ids=dl['token_type_ids'].to(device)
            mask= dl['mask'].to(device)
            label=dl['target'].to(device)
            label = label.unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            
            output=model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids)
            label = label.type_as(output)

            loss=loss_fn(output,label)
            loss.backward()
            
            optimizer.step()
            output =output.cpu().detach().numpy()
            pred = np.where(output >= 0, 1, 0)

            num_correct = sum(1 for a, b in zip(pred, label) if a[0] == b[0])
            num_samples = pred.shape[0]
            accuracy = num_correct/num_samples
        if epoch%2:
            print(f'Training:  {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
            validation_loop(val_dataloader,model)   
            # Show progress while training
            loop.set_description(f'Epoch={epoch}/{epochs}')
            loop.set_postfix(loss=loss.item(),acc=accuracy)

    return model

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer= optim.Adam(params,lr= args['LR'])
loss_fn = nn.BCEWithLogitsLoss()
model = training_loop(epochs, train_dataloader,val_dataloader, model, loss_fn, optimizer)
torch.save(model.state_dict(), 'ai-text-classifier.pth')



