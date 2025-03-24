# Data Preparation for Breakfast

## Download Annotations
Please download the annotation `data/annotations/` from [Box](https://oregonstate.box.com/s/fyxgm3a4vk5qtzygn2zot85ce48y2b9l).

The annotation should be in `data/annotations/`.

To train with varying FPS, adjust the `frame` and `fps` values in the JSON file, and ensure evaluation is performed at the correct FPS; previous work utilized 15 FPS.

## Download Raw Videos

Please put the downloaded video under the path: `data/breakfast/raw_data/video_fps3/`.

You can download the raw video from [official website](https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/), or download from this [Box](https://oregonstate.box.com/s/fyxgm3a4vk5qtzygn2zot85ce48y2b9l).

The videos in the Box are resized to a spatial resolution of 160x160 and downsampled to 3 FPS.

## Citation

```BibTeX
@inproceedings{Breakfast,
  title={The language of actions: Recovering the syntax and semantics of goal-directed human activities},
  author={Kuehne, Hilde and Arslan, Ali and Serre, Thomas},
  booktitle={CVPR},
  year={2014}
}
```