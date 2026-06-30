#!/usr/bin/python2.7

import torch,os
import numpy as np
import random
import pandas as pd

def normalize_last_dim(x, epsilon=1e-10):
    """
    Normalize along the last dimension, handling division by zero by using a small epsilon.
    
    Parameters:
    x : numpy.ndarray
        Input array.
    epsilon : float, optional
        Small constant to avoid division by zero (default is 1e-10).
    
    Returns:
    numpy.ndarray
        Array normalized along the last dimension.
    """
    # Sum along the last dimension
    sum_along_last = np.sum(x, axis=-1, keepdims=True)
    
    # Avoid division by zero: replace zero sums with epsilon
    sum_along_last = np.where(sum_along_last == 0, epsilon, sum_along_last)
    
    # Divide each element by the sum of the last dimension
    return x / sum_along_last

class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, directory_path, sample_rate, args):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        # self.gt_path = gt_path
        # self.features_path = features_path
        self.directory_path=directory_path
        self.sample_rate = sample_rate
        self.args=args

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        self.list_of_examples = [file_name for file_name in os.listdir(self.directory_path) if file_name.endswith('.npz')]
        random.shuffle(self.list_of_examples)
        # print("# location 2")
        # print(self.directory_path)
        # print(file_names)

    def next_batch(self, batch_size, dropped=20):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size
        N=30
        batch_input = []
        batch_target = []
        for vid in batch:
            file_path = os.path.join(self.directory_path, vid)
            data = np.load(file_path)
            df = pd.DataFrame(data['seg'], columns=['label', 'start', 'end', 'prob'])
            sorted_df = df.sort_values(by='prob', ascending=False)
            top_12_df = sorted_df.head(N)
            dropped_6_df = top_12_df.sample(n=min(dropped,N))
            sorted_df = sorted_df.drop(dropped_6_df.index)

            pred=np.zeros_like(data['prob'])
            for idx, this_pred in sorted_df.iterrows():
                label, s, e, score=this_pred[['label',"start", "end", 'prob']].values
                s,e,label=int(s),int(e),int(label)
                # print(label, s, e, score)
                pred[s:e, label]+=score
            features = normalize_last_dim(pred)
            classes = data['gt']
            # indices_not_minus_100 = np.where(classes != -100)[0]
            # classes=classes[indices_not_minus_100]
            # features=features[indices_not_minus_100]

            features=features.T

            batch_input.append(features[:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])

        length_of_sequences =  list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])
            # mask[i, :, indices_not_minus_100] = 1

        return batch_input_tensor, batch_target_tensor, mask
