import json, os, csv
import numpy as np
import pandas as pd
import multiprocessing as mp

from .builder import EVALUATORS, remove_duplicate_annotations
from collections import defaultdict
import math
from opentad.evaluations.mAP import segment_iou, k_segment_iou, interpolated_prec_rec

@EVALUATORS.register_module()
class mAP2_4:
    def __init__(
        self,
        ground_truth_filename,
        prediction_filename,
        subset,
        tiou_thresholds,
        nms_score_t=0.4,
        gt_rate=1.1,
        top_k=None,
        blocked_videos=None,
        thread=16,
        bg_class=[],
        work_dir='',
    ):
        super().__init__()
        # print("# location 10")
        # print(ground_truth_filename)
        # exit()

        assert work_dir != '', "The work_dir variable must not be an empty string."
        self.work_dir = work_dir
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

        # self.csv_video_path=work_dir+'/video_accuracy_edit_scores.csv'
        # with open(self.csv_video_path, mode='w', newline='') as file:
        #     header = ['name', 'acc', 'edit']
        #     writer = csv.writer(file)
        #     writer.writerow(header)

        if not ground_truth_filename:
            raise IOError("Please input a valid ground truth file.")
        if not prediction_filename:
            raise IOError("Please input a valid prediction file.")

        self.subset = subset
        self.tiou_thresholds = tiou_thresholds
        self.top_k = top_k
        self.gt_fields = ["database"]
        self.pred_fields = ["results"]
        self.thread = thread  # multi-process workers
        self.nms_score_t=nms_score_t
        self.bg_class = bg_class
        self.bg_idx = []

        # Get blocked videos
        if blocked_videos is None:
            self.blocked_videos = list()
        else:
            with open(blocked_videos) as json_file:
                self.blocked_videos = json.load(json_file)

        # Import ground truth and predictions.
        # print(ground_truth_filename)
        # exit()
        self.ground_truth, self.activity_index, self.activity_index_all, self.video_index, self.video_info = \
           self._import_ground_truth(ground_truth_filename)
        # print(self.activity_index)
        # exit()

        self.video_index_r = {v: k for k, v in self.video_index.items()}

        self.prediction = self._import_prediction(prediction_filename)
        # print(self.activity_index)
        return 

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        dataset=ground_truth_filename.split('/')[-1].split('.')[0]
        category_path=ground_truth_filename.split(dataset)[0]+dataset+'_category_idx.txt'
        # print("# location 1")
        # print(ground_truth_filename)
        # print(category_path)
        # print(self.bg_class)
        # return 
        activity_index, cidx = {}, 0
        with open(category_path, 'r') as file:
            lines = [line.strip() for line in file if line.strip()]
            # print(lines)
            for a in lines:
                activity_index[a]=cidx
                cidx+=1
        activity_index_ret={}
        # print(activity_index)
        
        with open(ground_truth_filename, "r") as fobj:
            data = json.load(fobj)

        # Checking format
        if not all([field in list(data.keys()) for field in self.gt_fields]):
            raise IOError("Please input a valid ground truth file.")

        # Read ground truth data.
        
        video_index, vidx = {}, 0
        video_info=defaultdict(dict)
        video_lst, t_start_lst, t_end_lst, label_lst, vlabel_lst = [], [], [], [], []
        for videoid, v in data["database"].items():
            if self.subset != v["subset"]:
                continue
            if videoid in self.blocked_videos:
                continue

            # remove duplicated instances following ActionFormer
            # print(v["annotations"])
            # exit()
            v_anno = remove_duplicate_annotations(v["annotations"])
            # print(v_anno)
            if videoid not in video_index:
                video_index[videoid] = vidx
                video_info[vidx]['duration']=v['duration']
                video_info[vidx]['frame']=v['frame']
                video_info[vidx]['name']=videoid
                vidx += 1
                # print("# location 2 ")
                # print(v,videoid)

                # exit()
            for ann in v_anno:
                if ann["label"] not in activity_index_ret:

                    activity_index_ret[ann["label"]] = activity_index[ann["label"]]
                if ann["label"] in self.bg_class:
                    self.bg_idx.append(activity_index[ann["label"]])
                    # cidx += 1
                video_lst.append(videoid)
                vlabel_lst.append(video_index[videoid])
                t_start_lst.append(float(ann["segment"][0]))
                t_end_lst.append(float(ann["segment"][1]))
                label_lst.append(activity_index[ann["label"]])

        ground_truth = pd.DataFrame(
            {
                "video-id": video_lst,
                "t-start": t_start_lst,
                "t-end": t_end_lst,
                "label": label_lst,
                "vlabel": vlabel_lst,
            }
        )
        return ground_truth, activity_index_ret, activity_index, video_index, video_info

    def _import_prediction(self, prediction_filename):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.

        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.

        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        # if prediction_filename is a string, then json load
        if isinstance(prediction_filename, str):
            with open(prediction_filename, "r") as fobj:
                data = json.load(fobj)
        elif isinstance(prediction_filename, dict):
            data = prediction_filename
        else:
            raise IOError(f"Type of prediction file is {type(prediction_filename)}.")

        # Checking format...
        if not all([field in list(data.keys()) for field in self.pred_fields]):
            raise IOError("Please input a valid prediction file.")

        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, vlabel_lst, score_lst = [], [], []
        for video_id, v in data["results"].items():
            if video_id in self.blocked_videos:
                continue
            for result in v:
                try:
                    label = self.activity_index_all[result["label"]]
                    vlabel = self.video_index[video_id]
                except:
                    # this is because the predicted label is not in annotation
                    # such as the some classes only exists in train split, but not in val split
                    label = len(self.activity_index_all)
                    vlabel = len(self.video_index)
                video_lst.append(video_id)
                t_start_lst.append(float(result["segment"][0]))
                t_end_lst.append(float(result["segment"][1]))
                label_lst.append(label)
                vlabel_lst.append(vlabel)
                score_lst.append(result["score"])
                # score_og_lst.append(result["score_og"])
                # print(result["score"])
        prediction = pd.DataFrame(
            {
                "video-id": video_lst,
                "t-start": t_start_lst,
                "t-end": t_end_lst,
                "label": label_lst,
                "score": score_lst,
                # "score_og": score_og_lst,
                "vlabel": vlabel_lst,
            }
        )
        return prediction

    def wrapper_compute_acc(self, vidx_list):
        """Computes average precision for a sub class list."""
        for vidx in vidx_list:
            gt_idx = self.ground_truth["vlabel"] == vidx
            pred_idx = self.prediction["vlabel"] == vidx
            self.acc_result_dict[vidx] = compute_acc(
                vidx,
                self.ground_truth.loc[gt_idx].reset_index(drop=True),
                self.prediction.loc[pred_idx].reset_index(drop=True),
                self.video_info,
                self.video_index_r,
                self.activity_index_all,
                self.nms_score_t,
                self.bg_idx,
                self.work_dir,
            )


    def wrapper_compute_average_precision(self, cidx_list):
        """Computes average precision for a sub class list."""
        # print(cidx_list)
        for cidx in cidx_list:
            gt_idx = self.ground_truth["label"] == cidx
            pred_idx = self.prediction["label"] == cidx
            # print(cidx,gt_idx,pred_idx)
            # print(self.ground_truth["label"])
            self.mAP_result_dict[cidx] = compute_average_precision_detection(
                self.ground_truth.loc[gt_idx].reset_index(drop=True),
                self.prediction.loc[pred_idx].reset_index(drop=True),
                tiou_thresholds=self.tiou_thresholds,
            )

    def wrapper_compute_topkx_recall(self, cidx_list):
        """Computes Top-kx recall for a sub class list."""
        for cidx in cidx_list:
            gt_idx = self.ground_truth["label"] == cidx
            pred_idx = self.prediction["label"] == cidx
            self.recall_result_dict[cidx] = compute_topkx_recall_detection(
                self.ground_truth.loc[gt_idx].reset_index(drop=True),
                self.prediction.loc[pred_idx].reset_index(drop=True),
                tiou_thresholds=self.tiou_thresholds,
                top_k=self.top_k,
            )
    def multi_thread_compute_acc(self):
        self.acc_result_dict = mp.Manager().dict()

        num_total = len(self.video_index.values())
        num_video_per_thread = num_total // self.thread + 1

        processes = []
        for tid in range(self.thread):
            num_start = int(tid * num_video_per_thread)
            num_end = min(num_start + num_video_per_thread, num_total)

            p = mp.Process(
                target=self.wrapper_compute_acc,
                args=(list(self.video_index.values())[num_start:num_end],),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        acc = np.zeros((3))
        means = np.zeros((10))
        for i, vidx in enumerate(self.video_index.values()):
            acc  += self.acc_result_dict[vidx]['acc']
            means += self.acc_result_dict[vidx]['means']
        return acc[0]/acc[1], acc[2]/len(self.video_index),means/len(self.video_index)


    def multi_thread_compute_average_precision(self):
        self.mAP_result_dict = mp.Manager().dict()

        num_total = len(self.activity_index.values())
        num_activity_per_thread = num_total // self.thread + 1
        # print(self.activity_index.values())
        processes = []
        for tid in range(self.thread):
            num_start = int(tid * num_activity_per_thread)
            num_end = min(num_start + num_activity_per_thread, num_total)

            p = mp.Process(
                target=self.wrapper_compute_average_precision,
                args=(list(self.activity_index.values())[num_start:num_end],),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index.items())))
        for i, cidx in enumerate(self.activity_index.values()):
            ap[:, i] = self.mAP_result_dict[cidx]
        return ap

    def multi_thread_compute_topkx_recall(self):
        self.recall_result_dict = mp.Manager().dict()

        num_total = len(self.activity_index.values())
        num_activity_per_thread = num_total // self.thread + 1

        processes = []
        for tid in range(self.thread):
            num_start = int(tid * num_activity_per_thread)
            num_end = min(num_start + num_activity_per_thread, num_total)

            p = mp.Process(
                target=self.wrapper_compute_topkx_recall,
                args=(list(self.activity_index.values())[num_start:num_end],),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        recall = np.zeros((len(self.tiou_thresholds), len(self.top_k), len(self.activity_index.items())))
        for i, cidx in enumerate(self.activity_index.values()):
            recall[..., i] = self.recall_result_dict[cidx]
        return recall

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """

        self.acc,self.edit, self.means = self.multi_thread_compute_acc()
        print("# location 1")
        print(self.means)

        self.ap = self.multi_thread_compute_average_precision()
        self.mAPs = self.ap.mean(axis=1)
        self.average_mAP = self.mAPs.mean()

        metric_dict = dict(average_mAP=self.average_mAP)
        for tiou, mAP in zip(self.tiou_thresholds, self.mAPs):
            metric_dict[f"mAP@{tiou}"] = mAP

        # if top_k is not None, we will compute top-kx recall
        if self.top_k is not None:
            self.recall = self.multi_thread_compute_topkx_recall()
            self.mRecall = self.recall.mean(axis=2)

            for tiou, mRecall in zip(self.tiou_thresholds, self.mRecall):
                for k, recall in zip(self.top_k, mRecall):
                    metric_dict[f"recall@{tiou}@{k}"] = recall
        metric_dict0={}
        for tiou, recall, p in zip(self.tiou_thresholds, self.mRecall, self.mAPs):
            r=recall[0]
            metric_dict0[f"F1@{tiou}"] = 2*p*r/(p+r)
        metric_dict0['Edit']=self.edit
        metric_dict0['Acc']=self.acc
        metric_dict0['nms_score_t']= self.nms_score_t
        return metric_dict0

    def logging(self, logger=None):
        if logger == None:
            pprint = print
        else:
            pprint = logger.info

        pprint("Loaded annotations from {} subset.".format(self.subset))
        pprint("Number of ground truth instances: {}".format(len(self.ground_truth)))
        pprint("Number of predictions: {}".format(len(self.prediction)))
        pprint("Fixed threshold for tiou score: {}".format(self.tiou_thresholds))
        pprint("Average-mAP: {:>4.2f} (%)".format(self.average_mAP * 100))
        pprint("Accuracy is {:>4.2f}%".format(self.acc * 100))
        pprint("Edit is {:>4.2f}%".format(self.edit*100))
        for tiou, mAP in zip(self.tiou_thresholds, self.mAPs):
            pprint("mAP at tIoU {:.2f} is {:>4.2f}%".format(tiou, mAP * 100))

        # if top_k is not None, print top-kx recall
        if self.top_k is not None:
            pprint("Fixed top-kx results: {}".format(self.top_k))
            for tiou, recall, p in zip(self.tiou_thresholds, self.mRecall, self.mAPs):
                recall_string = ["R{:d} is {:>4.2f}%".format(k, r * 100) for k, r in zip(self.top_k, recall)]
                # pprint("Recall at tIoU {:.2f}: {}".format(tiou, ", ".join(recall_string)))
                r=recall[0]
                F1=2*p*r/(p+r)
                pprint("F1 at tIoU {:.2f} is {:>4.2f}%".format(tiou, F1 * 100))

def levenstein(p, y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float64)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)
    
    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score

def segs_to_labels_start_end_time(seg_list, bg_class):
    seg_list = [ s for s in seg_list if s.action not in bg_class ]
    labels = [ p.action for p in seg_list ]
    start  = [ p.start for p in seg_list ]
    end    = [ p.end+1 for p in seg_list ]
    return labels, start, end

def edit_score(pred_segs, gt_segs, norm=True, bg_class=["background"]):
    P, _, _ = segs_to_labels_start_end_time(pred_segs, bg_class)
    Y, _, _ = segs_to_labels_start_end_time(gt_segs, bg_class)
    return levenstein(P, Y, norm)

def remove_duplicates(nums, bg_idx):
    if not nums:
        return []
    result=[]
    # print(bg_idx)
    nums = [item for index, item in enumerate(nums) 
                   if item not in bg_idx or index == 0 or index == len(nums) - 1]
    for i in range(len(nums)):
        # if nums[i] in bg_idx:continue
        # print(nums[i], end =' ')
        if len(result)==0:
            result = [nums[i]]
        elif nums[i] != result[-1]:    
            result.append(nums[i])
    return result
    # filtered_result = [item for index, item in enumerate(result) 
    #                if item not in bg_idx or index == 0 or index == len(result) - 1]
    # return filtered_result

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

def calculate_mean_for_lower_and_higher(arr, another_arr):
    """
    Calculate the mean values from another_arr for elements in arr
    that are below and above the median.
    
    Parameters:
    arr (numpy.ndarray): 1D array of numeric values.
    another_arr (numpy.ndarray): 1D array of boolean values.
    
    Returns:
    tuple: Mean values for lower and higher parts.
    """
    # Step 1: Find the median value of the array
    median_value = np.median(arr)
    
    # Step 2: Find the indices for lower and higher parts
    lower_indices = np.where(arr < median_value)[0]  # Values smaller than the median
    higher_indices = np.where(arr > median_value)[0]  # Values larger than the median
    
    # Step 3: Select values from another_arr corresponding to these indices
    lower_values = another_arr[lower_indices]
    higher_values = another_arr[higher_indices]
    
    # Step 4: Calculate mean of boolean values
    lower_mean = lower_values.mean() if len(lower_values) > 0 else 0
    higher_mean = higher_values.mean() if len(higher_values) > 0 else 0
    
    return lower_mean, higher_mean

def calculate_mean_in_10_parts(arr, another_arr):
    """
    Sort `another_arr` based on the permutation that sorts `arr`, 
    then divide it into 10 equal parts and return the mean of each part.
    
    Parameters:
    arr (numpy.ndarray): 1D array of numeric values.
    another_arr (numpy.ndarray): 1D array of boolean values.
    
    Returns:
    list: Mean of each of the 10 parts of the sorted `another_arr`.
    """
    # Step 1: Get the sorting permutation for `arr`
    sorted_indices = np.argsort(arr)
    
    # Step 2: Sort `another_arr` based on this permutation
    sorted_another_arr = another_arr[sorted_indices]
    
    # Step 3: Divide the sorted array into 10 equal parts
    num_parts = 10
    part_size = len(another_arr) // num_parts
    
    # Ensure that we can divide the array into exactly 10 parts
    if part_size == 0:
        raise ValueError("Array is too small to divide into 10 parts.")
    
    means = []
    
    # Step 4: Calculate the mean of each part
    for i in range(num_parts):
        # Handle the last part to include any remainder elements
        if i == num_parts - 1:
            part = sorted_another_arr[i * part_size:]
        else:
            part = sorted_another_arr[i * part_size : (i + 1) * part_size]
        
        means.append(np.mean(part))
    
    return means

def compute_acc(vidx, ground_truth, prediction, video_info, video_index_r,\
    activity_index, nms_score_t, bg_idx,work_dir):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    # print("# location 1 ")
    # print(video_info[vidx],vidx, work_dir)
    # return 


    duration, total_frame = video_info[vidx]['duration'], video_info[vidx]['frame']
    # print(duration,total_frame,video_info[vidx]['name'])
    # exit()
    total_frame = total_frame#*5
    rate = total_frame/duration
    # print(rate)
    # exit()
    gt=np.zeros((total_frame), dtype=int)-100
    num_classes=len(activity_index)
    pred=np.zeros((total_frame, num_classes))
    # print("# location 1.1")
    g_trans=[]
    # print(ground_truth)
    for idx, this_gt in ground_truth.iterrows():
        label,s,e=this_gt[['label',"t-start", "t-end"]].values
        # print(label,end=' ')
        s,e=int(s*rate),int(e*rate)
        gt[s:e]=label
        # print(label,s,e)
        g_trans.append(label)

    prediction['t-mid']= (prediction['t-start'] + prediction['t-end'])/2
    sorted_df = prediction.sort_values(by='t-mid')

    p_trans=[]
    for idx, this_pred in sorted_df.iterrows():
        label, s, e, score=this_pred[['label',"t-start", "t-end", 'score']].values
        if score<nms_score_t:continue
        p_trans.append(label)
    g_trans=remove_duplicates(g_trans, bg_idx)
    p_trans=remove_duplicates(p_trans, bg_idx)
    # print(g_trans)
    # print(p_trans)
    # exit()
    if len(bg_idx)!=0:
        if len(p_trans)==0:
            p_trans = [g_trans[0]] + p_trans
        elif p_trans[0] not in bg_idx and p_trans[0]!=g_trans[0]:
            p_trans = [g_trans[0]] + p_trans
        if p_trans[-1] not in bg_idx and p_trans[-1]!=g_trans[-1]:
            p_trans.append(g_trans[-1])
    # print(vidx)
   
    edit=levenstein(g_trans, p_trans, norm=True)/100

    for idx, this_pred in sorted_df.iterrows():
        label, s, e, score=this_pred[['label',"t-start", "t-end", 'score']].values
        s,e=int(s*rate),int(e*rate)
        pred[s:e, label]+=score

    pred = normalize_last_dim(pred)
    pred0 = np.argmax(pred, axis=1)
    pred1 = np.max(pred, axis=1)
    
    
    save_path=work_dir+'/'+video_info[vidx]['name']+'.npz'
    # print("# location 1")
    # print(sorted_df.values.shape)
    # # print(save_path)
    # exit(-1)
    sorted_df['t-start']=sorted_df['t-start']*rate
    sorted_df['t-end']=sorted_df['t-end']*rate

    
    ground_truth['t-start']=ground_truth['t-start']*rate
    ground_truth['t-end']=ground_truth['t-end']*rate

    np.savez(save_path, gt=gt, prob=pred, seg=sorted_df[['label',"t-start", "t-end", 'score']].values, \
        ground_truth=ground_truth[['label',"t-start", "t-end"]].values)

    # gt is gt
    # pred1 is the highest probability
    # pred0 is the prediction
    indices_not_minus_100 = np.where(gt != -100)[0]

    gt=gt[indices_not_minus_100]
    pred0=pred0[indices_not_minus_100]
    pred1=pred1[indices_not_minus_100]
    pred=pred[indices_not_minus_100]



    correct = (gt == pred0)
    correct_predictions = np.sum(correct)
    means =  calculate_mean_in_10_parts(pred1, correct)
    
    
    total_predictions = gt.shape[0]
    ret = {
        'acc':np.array([correct_predictions,total_predictions, edit]),
        'means':np.array(means),
    }

    return ret

def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction["score"].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby("video-id")

    # Assigning true positive to truly ground truth instances.
    for idx, this_pred in prediction.iterrows():
        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred["video-id"])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[["t-start", "t-end"]].values, this_gt[["t-start", "t-end"]].values)

        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]["index"]] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]["index"]] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    ap = np.zeros(len(tiou_thresholds))

    for tidx in range(len(tiou_thresholds)):
        # Computing prec-rec
        this_tp = np.cumsum(tp[tidx, :]).astype(float)
        this_fp = np.cumsum(fp[tidx, :]).astype(float)
        rec = this_tp / npos
        prec = this_tp / (this_tp + this_fp)
        ap[tidx] = interpolated_prec_rec(prec, rec)
    return ap


def compute_topkx_recall_detection(
    ground_truth,
    prediction,
    tiou_thresholds=np.linspace(0.1, 0.5, 5),
    top_k=(1, 5),
):
    """Compute recall (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    top_k: tuple, optional
        Top-kx results of a action category where x stands for the number of
        instances for the action category in the video.
    Outputs
    -------
    recall : float
        Recall score.
    """
    if prediction.empty:
        return np.zeros((len(tiou_thresholds), len(top_k)))

    # Initialize true positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(top_k)))
    n_gts = 0

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby("video-id")
    prediction_gbvn = prediction.groupby("video-id")

    for videoid, _ in ground_truth_gbvn.groups.items():
        ground_truth_videoid = ground_truth_gbvn.get_group(videoid)
        n_gts += len(ground_truth_videoid)
        try:
            prediction_videoid = prediction_gbvn.get_group(videoid)
        except Exception as e:
            continue

        this_gt = ground_truth_videoid.reset_index()
        this_pred = prediction_videoid.reset_index()

        # Sort predictions by decreasing score order.
        score_sort_idx = this_pred["score"].values.argsort()[::-1]
        top_kx_idx = score_sort_idx[: max(top_k) * len(this_gt)]
        tiou_arr = k_segment_iou(
            this_pred[["t-start", "t-end"]].values[top_kx_idx], this_gt[["t-start", "t-end"]].values
        )

        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for kidx, k in enumerate(top_k):
                tiou = tiou_arr[: k * len(this_gt)]
                tp[tidx, kidx] += ((tiou >= tiou_thr).sum(axis=0) > 0).sum()

    recall = tp / n_gts

    return recall
