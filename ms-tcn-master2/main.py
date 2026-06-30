#!/usr/bin/python2.7

import torch
from model import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random
import logging


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="gtea")
parser.add_argument('--split', default='1')
parser.add_argument('--sampler', default='poisson', type=str)
parser.add_argument('--lambda_val', default=5, type=int)
parser.add_argument(
    '--bg_idx', 
    nargs='*',  # "*" means zero or more arguments are expected
    type=int,   # Ensure the input is treated as integers
    default=[],  # Default to an empty list if no arguments are provided
    help='edit score'
)
parser.add_argument('--vis', default=0, type=int)

parser.add_argument('--directory_path', type=str,  
        default='exps/gtea/adatad/e2e_actionformer_ret_ssv2_tpl_l16_25m_768x1_160_adapter2_6/split1/gpu2_id0/')
parser.add_argument('--directory_path_eva', type=str,  
        default='exps/gtea/adatad/e2e_actionformer_ret_ssv2_tpl_l16_25m_768x1_160_adapter2_4/split1/gpu2_id0/')
parser.add_argument('--directory_path_eva1', type=str, default='')

args = parser.parse_args()

num_stages = 3
num_layers = 10
num_f_maps = 64
bz = 1
lr = 0.0005
num_epochs = 100

# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
# if args.dataset == "50salads":
#     sample_rate = 2

# vid_list_file = "./data/"+args.dataset+"/splits/train.split"+args.split+".bundle"
# vid_list_file_tst = "./data/"+args.dataset+"/splits/test.split"+args.split+".bundle"
# features_path = "./data/"+args.dataset+"/features/"
# gt_path = "./data/"+args.dataset+"/groundTruth/"

mapping_file = "./data/"+args.dataset+"/mapping.txt"

model_dir = "./models/"+args.dataset+"/l_{}{}/split_{}".format(args.lambda_val,args.sampler,args.split)+'/'
results_dir = "./results/"+args.dataset+"/split_"+args.split+'/'
args.results_dir=results_dir
# print(results_dir)
 
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

switched_dict = {v: k for k, v in actions_dict.items()}

num_classes = len(actions_dict)
features_dim = num_classes
args.device=device

logger = logging.getLogger('training_logger')
logger.setLevel(logging.INFO)

# Create handlers
file_handler = logging.FileHandler(model_dir+'training.log')  # Log to file
console_handler = logging.StreamHandler()  # Log to terminal

# Set formatters
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

trainer = Trainer(num_stages, num_layers, num_f_maps, features_dim, num_classes,args)
if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, args.directory_path, sample_rate,args)
    batch_gen.read_data(args.directory_path)
    trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)

if args.action == "predict":
    trainer.predict(model_dir+'best_eva_FEA.model', device)

if args.action == "predict1":
    trainer.predict1(model_dir+'best_eva_FEA.model', device)