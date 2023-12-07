import os
import json
import random
from collections import defaultdict

import torch
from torch.optim import AdamW
from torch.quasirandom import SobolEngine
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import *
from info import *
from util import *
from model import *


class Args:
    def __init__(self, tuning_model='', soft_token=0):
        self.tuning_model = tuning_model
        self.soft_token = soft_token

tuning_model = 'llama-30b'
soft_token = 5
args = Args(tuning_model, soft_token)
info = Info(args)

FILE_SOFT = os.path.join(info.DIR_SOFT, f'{args.tuning_model}-{args.soft_token}_.pkl') ##########
myprint(time.strftime("%Y %b %d %a, %H:%M:%S: ", time.localtime()) + 'Start Learning the Soft Prefix Instructions', FILE_SOFT.replace('pkl', 'txt'))

datasets_all = {dataset.split('.')[0]: Dataset(info, dataset.split('.')[0]) for dataset in os.listdir(info.DIR_INPUT)}
instructions = json.load(open(info.FILE_INSTRUCTION, 'r'))
instructions = [(dataset, hard) for dataset_instructions_ in instructions.values() for dataset, instructions_ in dataset_instructions_.items() for hard in instructions_]
# source_datasets = ['active_to_passive', 'translation_en-de', 'larger_animal', 'sentiment', 'singular_to_plural']
# instructions = [(dataset, hard) for dataset_instructions_ in instructions.values() for dataset in source_datasets for hard in dataset_instructions_[dataset]]

tuning_model_name = info.model2name[args.tuning_model]
tokenizer_ = AutoTokenizer.from_pretrained(tuning_model_name, use_fast=False)
model_ = AutoModelForCausalLM.from_pretrained(tuning_model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map='auto')
tuning_model = Model_HF(tokenizer_, model_)

for param in tuning_model.model.parameters():
    param.requires_grad = False
    
soft_lr = 0.005
soft_epoch = 100
soft_update_freq = 5

dim = tuning_model.model.get_input_embeddings().weight.shape[1]
softs = SobolEngine(dimension=dim, scramble=True).draw(len(instructions) * args.soft_token)
softs = softs.reshape(len(instructions), args.soft_token, -1)

all_softs = {}
report_each = len(instructions) // 50
for idx, (dataset_, hard_) in enumerate(instructions):
    
    dataset = datasets_all[dataset_]
    task_type = dataset.task_type
    
    num_batch = math.ceil(len(dataset.splits['train']) / info.num_demo)
    update_freq = min(num_batch, soft_update_freq)
    
    soft_ = nn.Parameter(softs[idx].to(info.DEVICE_GPU))
    
    if dataset_ not in ['cosmos_qa', 'hellaswag', 'imdb', 'anli', 'boolq', 'ag_news']:
        optimizer = AdamW([soft_], lr=soft_lr)
        optimizer.zero_grad()

        for idx_epoch in range(soft_epoch):
            random.shuffle(dataset.splits['train'])

            for idx_batch in range(num_batch):
                examples = dataset.splits['train'][idx_batch*info.num_demo : (idx_batch+1)*info.num_demo]
                loss = tuning_model.discover_instruction_prepend(examples, soft_, hard_, task_type)
                (loss / update_freq).backward()

                if (idx_batch + 1) % update_freq == 0 or idx_batch + 1 == num_batch:
                    optimizer.step()
                    optimizer.zero_grad()
                
    all_softs[(dataset_, hard_)] = soft_.detach().cpu().numpy().flatten()
    if idx % report_each == 0:
        pk.dump(all_softs, open(FILE_SOFT, 'wb'), -1)
        myprint(f'({idx}) {dataset_}: {hard_}', FILE_SOFT.replace('pkl', 'txt'))
pk.dump(all_softs, open(FILE_SOFT, 'wb'), -1)