import csv
from models.bert import AITextDetector
from models.roberta_model import RobertaAITextDetector
from utils.dataloader import AITextDataset
from tqdm import tqdm
import transformers
import torch
import argparse
import yaml
import numpy as np
from torch.utils.data import DataLoader



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/test.cfg',
                        help='config file. see readme')
    parser.add_argument('--data', type=str, default='test',
                        help='Specify the dataset type : train/test')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum length')
    parser.add_argument('--arch', type=str, default='bert',
                        help='Specify the arch type : bert/roberta')

    return parser.parse_args()

args = parse_args()
dataset_type = args.data
max_length = args.max_length
arch = args.arch


with open(args.cfg, 'r') as f:
        args = yaml.load(f, Loader=yaml.Loader)

torch.manual_seed(args['SEED'])
torch.backends.cudnn.deterministic = args['DETERMINISTIC']
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if arch =='bert':
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
else:
    tokenizer =transformers.RobertaTokenizer.from_pretrained("roberta-base")
if arch =='bert':
    model = AITextDetector()
else:
    model = RobertaAITextDetector()


dataset = AITextDataset(tokenizer =tokenizer, max_length = max_length, data_type = dataset_type,model_type=arch, file_path =args['PATH'])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args['BATCH_SIZE'], shuffle=False,collate_fn=dataset.collate_fn_test)
model.load_state_dict(torch.load('new_batch/ai-text-classifier-79.0.pth'))
model.eval()
model.to(device)
# _  = validation_loop(dataloader,model)
with torch.no_grad():
    with open("submission-16.csv", "w") as file:
        csv_out = csv.writer(file)
        csv_out.writerow(['id','label'])
        loop=tqdm(enumerate(dataloader),leave=False,total=len(dataloader))

        for i,(batch, dl) in enumerate(loop):
            ids = dl['ids'].to(device)
            if arch =='bert':
                token_type_ids=dl['token_type_ids'].to(device)
            mask= dl['mask'].to(device)
           
            if arch =='bert':
                output = model(ids,mask,token_type_ids)
            else:
                 output=model(ids,mask)
            output =output.cpu().detach().numpy()
            pred = np.where(output >= 0, 1, 0)  
            row = pred.tolist()[0][0]
            csv_out.writerow([i, row])
 


