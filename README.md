# EAST: End-to-End Action Segmentation Transformer

<p align="left">
<a href="https://arxiv.org/abs/2503.06316" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2503.06316-b31b1b.svg?style=flat" /></a>
</p>

## 🌟 Model Zoo

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <ul>
            <a href="configs/adatad/gtea">GTEA</a>
        </ul>
      </td>
      <td>
         <ul>
            <a href="configs/adatad/50salads">50salads</a>
        </ul>
      </td>
      <td>
         <ul>
            <a href="configs/adatad/breakfast">Breakfast</a>
        </ul>
      </td>
      <td>
         <ul>
            <a href="configs/adatad/assembly101">Assembly101</a>
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

The detailed configs, results, and pretrained models of each method can be found in above folders.

## 🛠️ Installation
Please refer to [install.md](docs/en/install.md) for installation.

## 📝 Data Preparation

Please refer to [data.md](docs/en/data.md) for data preparation.

## 🚀 Usage

Please refer to [usage.md](docs/en/usage.md) for details of training and evaluation scripts.

EAST follows a two-stage *segmentation-by-detection* pipeline:

1. **Detection** — the EAST transformer detector predicts action proposals over a coarsely down-sampled video (`tools/train.py` / `tools/test.py`; configs under `configs/adatad/`).
2. **High-Frame-Rate Aggregation and Refinement** — the proposals are aggregated and refined by an MS-TCN to produce the final frame-wise segmentation (Sec. 3.3–3.5 of the paper). See [`ms-tcn-master2/`](ms-tcn-master2/README.md).

## 🖊️ Citation

**[Acknowledgement]** This repo is inspired by [OpenTAD](https://github.com/sming256/OpenTAD/tree/main) project, and we give our thanks to their contributors. The refinement stage in [`ms-tcn-master2/`](ms-tcn-master2/README.md) is derived from [MS-TCN](https://github.com/yabufarha/ms-tcn); we thank its authors as well.

If you think this repo is helpful, please cite us:

```bibtex
@article{wang2025end,
  title={End-to-End Action Segmentation Transformer},
  author={Wang, Tieqiao and Todorovic, Sinisa},
  journal={arXiv preprint arXiv:2503.06316},
  year={2025}
}
```

If you have any questions, please contact: `wangtie@oregonstate.edu`.