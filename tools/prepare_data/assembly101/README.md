# Data Preparation for Assembly101

## Download Annotations
Please download the annotation `data/annotations/` from [Box](https://oregonstate.box.com/s/fyxgm3a4vk5qtzygn2zot85ce48y2b9l).

The annotation should be in `data/annotations/`.

To train with varying FPS, adjust the `frame` and `fps` values in the JSON file, and ensure evaluation is performed at the correct FPS; previous work utilized 15 FPS.

## Download Raw Videos

Please put the downloaded video under the path: `data/assembly101/Assembly104_4/`.

You can download the raw video from [official website](https://assembly-101.github.io/), or download from this [Box](https://oregonstate.box.com/s/fyxgm3a4vk5qtzygn2zot85ce48y2b9l).

The videos in the Box are cropped and resized to a spatial resolution of 160x160 and downsampled to 6 FPS.

## Citation

```BibTeX
@inproceedings{Assembly101,
  title={Assembly101: A Large-Scale Multi-View Video Dataset for Understanding Procedural Activities},
  author={Sener, Fadime and Chatterjee, Dibyadip and Shelepov, Daniel and He, Kun and Singhania, Dipika and Wang, Robert and Yao, Angela},
  booktitle={CVPR},
  year={2022}
}
```