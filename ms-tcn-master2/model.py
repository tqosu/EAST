#!/usr/bin/python2.7
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy,os
import numpy as np
import pandas as pd
from batch_gen import  normalize_last_dim 

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import json
from concurrent.futures import ProcessPoolExecutor
from scipy.special import softmax
from matplotlib.colors import ListedColormap, BoundaryNorm
import multiprocessing
import random
# other_module.py
import logging
from eval import edit_score,f_score,levenstein

# Get the logger configured in the main script
logger = logging.getLogger('training_logger')

import csv

def plot_gt_pred(gt, pred, name, save_path):
    # Ensure gt and pred have the same length
    if len(gt) != len(pred):
        raise ValueError("Ground truth and prediction arrays must have the same length.")
    
    # Determine unique classes and create a color map based on them
    unique_classes = np.unique(np.concatenate((gt, pred)))
    num_classes = len(unique_classes)
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, num_classes))  # Generate colors for each class
    cmap = ListedColormap(colors)
    
    # Create a mapping from class to color index
    class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
    
    # Map gt and pred values to the colormap indices
    gt_mapped = np.vectorize(class_to_index.get)(gt)
    pred_mapped = np.vectorize(class_to_index.get)(pred)
    
    # Create a boundary norm to handle color scaling for discrete values
    norm = BoundaryNorm(np.arange(num_classes + 1) - 0.5, num_classes)

    # Create the plot with two rows, one for gt and one for pred
    fig, axs = plt.subplots(nrows=2, figsize=(10, 4), constrained_layout=True)
    
    # Plot gt in the first row
    axs[0].imshow([gt_mapped], aspect='auto', cmap=cmap, norm=norm, interpolation='none')
    axs[0].axis('off')  # Hide axes for a cleaner look
    
    # Plot pred in the second row
    axs[1].imshow([pred_mapped], aspect='auto', cmap=cmap, norm=norm, interpolation='none')
    axs[1].axis('off')  # Hide axes for a cleaner look

    # Set a title for the entire plot using the name
    fig.suptitle(name, fontsize=16)

    # Save the plot as an image file
    file_path = f"{save_path}/{name}.png"
    plt.savefig(file_path, dpi=300)
    print(f"Plot saved at {file_path}")
    plt.close(fig)  # Close the figure to free memory

def insert_row_into_csv(file_name, row_values):
    # Define the header
    header = ['dataset', 'split', 'sampler', 'lambda', 'epoch', 'F1@10', 'F1@25', 'F1@50','Edit','f_Acc','o_Acc','FEA']
    
    # Check if the file exists
    file_exists = os.path.isfile(file_name)

    # Open the file in append mode
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header if the file does not exist
        if not file_exists:
            writer.writerow(header)
        
        # Insert the row values
        writer.writerow(row_values)
def remove_duplicates(nums, bg_idx):
    if nums.shape[0]==0:
        return []
    result=[]
    # print(bg_idx)
    nums = [item for index, item in enumerate(nums) 
                   if item not in bg_idx]
    for i in range(len(nums)):
        # if nums[i] in bg_idx:continue
        # print(nums[i], end =' ')
        if len(result)==0:
            result = [nums[i]]
        elif nums[i] != result[-1]:    
            result.append(nums[i])
    return result

class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]

def fill_to_length(arr, T):
    # Create an array of indices for the new length
    indices = np.linspace(0, len(arr) - 1, T)
    
    # Use nearest neighbor to fill the new array
    filled_values = arr[np.round(indices).astype(int)]
    
    return filled_values

class Trainer:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes, args):
        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        self.args=args

    def process_file(self, vid):
        device= self.args.device
        file_path = os.path.join(self.args.directory_path_eva, vid)
        data = np.load(file_path)
        features = normalize_last_dim(data['prob'])
        gt = data['gt']
        if self.args.directory_path_eva1 != '':
            file_path1 = os.path.join(self.args.directory_path_eva1, vid)
            data1 = np.load(file_path1)
            gt = data1['gt']
        
        print(len(gt),features.shape,vid)
        print(data['ground_truth'])
        # exit()
        
        # gt=gt[indices_not_minus_100]
        # features=features[indices_not_minus_100]

        input_x = torch.tensor(features.T, dtype=torch.float)
        input_x.unsqueeze_(0)
        input_x = input_x.to(device)
        predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
        _, predicted = torch.max(predictions[-1].data, 1)

        predicted0=np.argmax(data['prob'],axis=1)
        predicted = predicted.squeeze().cpu().numpy()

        # indices_not_minus_100 = np.where(gt != -100)[0]
        # print(len(gt))
        # print(gt)
        # gt=gt[indices_not_minus_100]
        # predicted0=predicted0[indices_not_minus_100]
        # predicted=predicted[indices_not_minus_100]
        # print(gt)
        predicted = fill_to_length(predicted, len(gt))
        predicted0 = fill_to_length(predicted0, len(gt))
        # print(len(gt),len(predicted))
        # exit()
        
        if self.args.vis==1:
            # print(gt)
            # print(predicted)
            plot_gt_pred(gt, predicted, vid.split('.')[0], self.args.results_dir)

        correct0=sum(predicted0==gt)
        correct=sum(predicted==gt)
        total=gt.shape[0]
        # print(correct0,correct,total)

        # ------------------------------

        overlap = [.1, .25, .5]
        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

        
        edit = 0
        gt_content=gt
        recog_content=predicted
        # print("# location 1 ")
        # print(list(gt))
        # print(list(predicted))
        # exit()
        
        # print("# location 2.1")
        # print(correct0,correct,total)
        bg_idx=self.args.bg_idx
        p_trans=remove_duplicates(predicted, bg_idx)
        g_trans=remove_duplicates(gt, bg_idx)
        
        # print(g_trans)
        # print(p_trans)
        # if len(bg_idx)!=0:
        #     if len(p_trans)==0:
        #         p_trans = [g_trans[0]] + p_trans
        #     elif p_trans[0] not in bg_idx and p_trans[0]!=g_trans[0]:
        #         p_trans = [g_trans[0]] + p_trans
        #     if p_trans[-1] not in bg_idx and p_trans[-1]!=g_trans[-1]:
        #         p_trans.append(g_trans[-1])
        edit += levenstein(g_trans, p_trans, norm=True)

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s],bg_idx)
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
        # print(correct0,correct,total)
        # exit()
        return [correct0,correct,total,edit,tp,fp,fn]

    def train_eva(self):
        file_names = [file_name for file_name in os.listdir(self.args.directory_path_eva) if file_name.endswith('.npz')]
        self.model.eval()
        results=[]
        # for vid in file_names:
        #     results.append(self.process_file(vid))
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(tqdm(executor.map(lambda vid: self.process_file(vid), file_names), 
                                total=len(file_names), desc="Processing videos"))
        overlap = [.1, .25, .5]
        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

        correct0 = 0
        correct = 0
        total = 0
        edit = 0
        
        for result in results:
            correct0 += result[0]
            correct += result[1]
            total += result[2]
            edit += result[3]
            # print("# location 2")
            # print(correct0,correct,total)
            # print(result[4])
            
            tp += result[4]
            fp += result[5]
            fn += result[6]

        ans=[]
        for s in range(len(overlap)):
            precision = tp[s] / float(tp[s]+fp[s])
            recall = tp[s] / float(tp[s]+fn[s])
        
            f1 = 2.0 * (precision*recall) / (precision+recall)

            f1 = np.nan_to_num(f1)*100

            # count *= 100
            # print ('F1@%0.2f: %.4f' % (overlap[s], f1))
            ans.append(f1)
        
        ans.append((1.0*edit)/len(results))
        # print ('Edit: %.4f' % ans[-1])
        ans.append(100*float(correct)/total)
        # print ("Acc: %.4f" % ans[-1])
        ans.append(100*float(correct0)/total)

        self.model.train()
        return np.array(ans)

    def process_file1(self, vid):
        device= self.args.device
        file_path = os.path.join(self.args.directory_path_eva, vid)
        data = np.load(file_path)

        features = normalize_last_dim(data['prob'])
        gt = data['gt']

        
        # gt=gt[indices_not_minus_100]
        # features=features[indices_not_minus_100]

        input_x = torch.tensor(features.T, dtype=torch.float)
        input_x.unsqueeze_(0)
        input_x = input_x.to(device)
        predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
        _, predicted = torch.max(predictions[-1].data, 1)

        predicted0=np.argmax(data['prob'],axis=1)
        predicted = predicted.squeeze().cpu().numpy()

        indices_not_minus_100 = np.where(gt != -100)[0]
        npz_path=self.args.results_dir + '/' + vid
        print(npz_path)
        data1={'gt':gt,'pred':predicted}
        np.savez(npz_path,**data1)
        # print(self.args.results_dir)
        # exit()
        # print(len(gt),predicted.shape,vid)
        # print(type(predicted),type(gt))
        # exit()
        
    
    def train_eva1(self):
        file_names = [file_name for file_name in os.listdir(self.args.directory_path_eva) if file_name.endswith('.npz')]
        self.model.eval()
        for vid in file_names:
            self.process_file1(vid)

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        best_acc=0
        best_FEA=0
        total_examples = len(batch_gen.list_of_examples)
        
        # Calculate the total number of batches per epoch and for all epochs
        total_batches_per_epoch = (total_examples + batch_size - 1) // batch_size
        total_batches_all_epochs = total_batches_per_epoch * num_epochs
        overall_progress_bar = tqdm(total=total_batches_all_epochs, desc='Overall Training Progress', unit='batch')

        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            # print("# location 1.1")
            cnt=0
            # with tqdm(total=total_batches, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as progress_bar:
            with tqdm(total=total_batches_per_epoch, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch', leave=False) as epoch_progress_bar:
                while batch_gen.has_next():
                    # print(cnt)
                    cnt+=1
                    if self.args.sampler=='poisson':
                        batch_input, batch_target, mask = batch_gen.next_batch(batch_size, np.random.poisson(self.args.lambda_val))
                    elif self.args.sampler=='uniform':
                        start=min(1,self.args.lambda_val)
                        batch_input, batch_target, mask = batch_gen.next_batch(batch_size, random.randint(start, self.args.lambda_val))
                    batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                    # print(' #  location 1')
                    # print(batch_input.shape,batch_target.shape,mask.shape)
                    
                    # return 
                    optimizer.zero_grad()
                    predictions = self.model(batch_input, mask)

                    loss = 0
                    for p in predictions:
                        loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                        loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                    epoch_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                    _, predicted = torch.max(predictions[-1].data, 1)
                    correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                    total += torch.sum(mask[:, 0, :]).item()
                    epoch_progress_bar.update(1)
                    overall_progress_bar.update(1)
            eva_acc=self.train_eva()
            cur_FEA=sum(eva_acc[:-1])/len(eva_acc[:-1])
            if eva_acc[-2]>best_acc:
                # print("[epoch %d]: best_eva_acc = %f" % (epoch + 1, eva_acc))
                logger.info("[epoch %d]: o_acc = %f best_eva_acc = %f" % (epoch + 1, eva_acc[-1], eva_acc[-2]))
                torch.save(self.model.state_dict(), save_dir + "/best_eva_acc.model")
                torch.save(optimizer.state_dict(), save_dir + "/best_eva_acc.opt") 
                best_acc=eva_acc[-2]
                row=[self.args.dataset, self.args.split, self.args.sampler, self.args.lambda_val, epoch+1]
                row.extend(eva_acc)
                row.append(cur_FEA)
                insert_row_into_csv('results.csv', row)
    
            if cur_FEA>best_FEA:
                logger.info("[epoch %d]: best_FEA = %f" % (epoch + 1, cur_FEA))
                torch.save(self.model.state_dict(), save_dir + "/best_eva_FEA.model")
                torch.save(optimizer.state_dict(), save_dir + "/best_eva_FEA.opt") 
                best_FEA=cur_FEA
                row=[self.args.dataset, self.args.split, self.args.sampler, self.args.lambda_val, epoch+1]
                row.extend(eva_acc)
                row.append(cur_FEA)
                insert_row_into_csv('results.csv', row)
                

            batch_gen.reset()

            logger.info("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples), float(correct)/total))
            # print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
            #                                                    float(correct)/total))
        overall_progress_bar.close()

    def predict(self, model_dir, device, epoch=-1):
        # print(model_dir)
        self.model.to(device)
        self.model.eval()
        self.model.load_state_dict(torch.load(model_dir))
        # self.train_eva1()
        eva_acc=self.train_eva()
        
        row=[self.args.dataset, self.args.split,self.args.sampler, self.args.lambda_val, epoch]
        row.extend(eva_acc)
        cur_FEA=sum(eva_acc[:-1])/len(eva_acc[:-1])
        row.append(cur_FEA)
        print(row)
        insert_row_into_csv('results_all.csv',row )
        # return 

    def predict1(self, model_dir, device, epoch=-1):
        # print(model_dir)
        # exit()
        self.model.to(device)
        self.model.eval()
        self.model.load_state_dict(torch.load(model_dir))
        self.train_eva1()
        # eva_acc=
        
        # row=[self.args.dataset, self.args.split, self.args.lambda_val, epoch]
        # row.extend(eva_acc)
        # print(row)
        # insert_row_into_csv('results_all.csv',row )
        return 
      