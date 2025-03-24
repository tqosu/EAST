# Data Preparation for 50Salads

## Download Annotations
Please download the annotation `data/annotations/` from [Box](https://oregonstate.box.com/s/fyxgm3a4vk5qtzygn2zot85ce48y2b9l).

The annotation should be in `data/annotations/`.

To train with varying FPS, adjust the `frame` and `fps` values in the JSON file, and ensure evaluation is performed at the correct FPS; previous work utilized 15 FPS.

## Download Raw Videos

Please put the downloaded video under the path: `data/50salads/raw_data/video_fps2/`.

You can download the raw video from [official website](https://discovery.dundee.ac.uk/en/datasets/50-salads), or download from this [Box](https://oregonstate.box.com/s/fyxgm3a4vk5qtzygn2zot85ce48y2b9l).

The videos in the Box are resized to a spatial resolution of 160x160 and downsampled to 2 FPS.

## Citation

```BibTeX
@inproceedings{50Salads,
  title={Combining embedded accelerometers with computer vision for recognizing food preparation activities},
  author={Stein, Sebastian and McKenna, Stephen J},
  booktitle={Proceedings of the 2013 ACM international joint conference on Pervasive and ubiquitous computing},
  year={2013}
}
```