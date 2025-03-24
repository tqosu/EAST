#!/usr/bin/env python
# This script will merge all the joson files into one, used for instance (panoptic) segmentation

import os
import json
from collections import defaultdict
import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="gtea", help='three dataset: breakfast, 50salads, gtea')
parser.add_argument('--split', default='1')
parser.add_argument('--path1', default='./TAD/')
parser.add_argument('--ext', default='.mp4')

args = parser.parse_args()

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
def get_video_properties(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    # Close the video file
    cap.release()

    return {"duration": duration, "frame": total_frames, "fps":fps}

def format_string_segments(strings,fps):
    # List to hold the formatted output
    formatted_list = []
    
    # Track the start of the current segment
    start_index = 0
    
    # Loop through the list and compare current string to the next one
    while start_index < len(strings):
        current_string = strings[start_index]
        end_index = start_index + 1
        
        # Keep incrementing the end_index while the next string is the same
        while end_index < len(strings) and strings[end_index] == current_string:
            end_index += 1
        
        # Append the start index, exclusive end index, and the current string
        if current_string!='background':
            formatted_list.append({'segment':[start_index/fps, end_index/fps],'label':current_string})
        
        # Update start_index to begin a new segment
        start_index = end_index
    
    return formatted_list
def save_list_to_txt(strings, filename):
    # Open the file in write mode
    with open(filename, 'w') as file:
        for string in strings:
            # Write each string to a new line
            file.write(string + '\n')
def main():
    path1=args.path1
    check_mkdir(path1)

    gt_path = "./data/"+args.dataset+"/groundTruth/"

    mapping_file = "./data/"+args.dataset+"/mapping.txt"
    vid_prefix='./data/{}/video/'.format(args.dataset)
    
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    catagory_idx=[]
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
        catagory_idx.append(a.split()[1])
    catagory_idx_path=path1+'{}_category_idx.txt'.format(args.dataset)
    print(catagory_idx_path)
    save_list_to_txt(catagory_idx,catagory_idx_path)
    
    database={}
    mymap={'train':'training','test':'validation'}
    for task in ['train','test']:
        vid_list_file = "./data/"+args.dataset+"/splits/{}.split".format(task)+args.split+".bundle"
        file_ptr = open(vid_list_file, 'r')
        list_of_examples = file_ptr.read().split('\n')[:-1]
        print(vid_list_file,len(list_of_examples))
        
        # string_list = ["hello", "hello", "world", "python", "python", "python"]
        # formatted = format_string_segments(string_list)
        # print(formatted)
        # return 
        for vid in list_of_examples:
            video_path=vid_prefix+vid.replace('.txt',args.ext)
            # if video_path.find('stereo')!=-1 and video_path.find('39')!=-1:
            print(video_path)


    #         # return 
            properties = get_video_properties(video_path)
            properties['subset']=mymap[task]
            # print(properties)

            file_ptr = open(gt_path + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]
            # print(properties,len(content))
            # continue
            properties['annotations']=format_string_segments(content,properties['fps'])
            name=vid.split('.')[0]
            database[name]=properties
            # print(properties)
    # return     
    filename = path1+args.dataset+".split{}".format(args.split)+".json"
    with open(filename, 'w') as file:
        json.dump({'database':database}, file, indent=4)
    
if __name__ == "__main__":
    main()
