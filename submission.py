import csv
from models.bert import AITextDetector
from utils.dataloader import AITextDataset
from tqdm import tqdm
import transformers
import torch
import argparse
import yaml
import numpy as np
from torch.utils.data import DataLoader
from train import validation_loop


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/test.cfg',
                        help='config file. see readme')
    parser.add_argument('--data', type=str, default='test',
                        help='Specify the dataset type : train/test')
    parser.add_argument('--max_length', type=int, default=300,
                        help='Maximum length')

    return parser.parse_args()

args = parse_args()
dataset_type = args.data
max_length = args.max_length

with open(args.cfg, 'r') as f:
        args = yaml.load(f, Loader=yaml.Loader)

torch.manual_seed(args['SEED'])
torch.backends.cudnn.deterministic = args['DETERMINISTIC']
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


model = AITextDetector(dropout = args['DROPOUT'])
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

dataset = AITextDataset(tokenizer =tokenizer, max_length = max_length, data_type = dataset_type,file_path =args['PATH'])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args['BATCH_SIZE'], shuffle=False)
model.load_state_dict(torch.load('new_train/ai-text-classifier-73.7.pth'))
model.eval()
model.to(device)
_  = validation_loop(dataloader,model)
with open("submission-6.csv", "w") as file:
    csv_out = csv.writer(file)
    csv_out.writerow(['id','label'])
    loop=tqdm(enumerate(dataloader),leave=False,total=len(dataloader))

    for i,(batch, dl) in enumerate(loop):
        ids=dl['ids'].to(device)
        token_type_ids=dl['token_type_ids'].to(device)
        mask= dl['mask'].to(device)
        output=model(
                    ids=ids,
                    mask=mask,
                    token_type_ids=token_type_ids)
        output =output.cpu().detach().numpy()
        pred = np.where(output >= 0, 1, 0)  
        row = pred.tolist()[0][0]
        csv_out.writerow([i, row])
 


