# Group Contextualization for Video Recognition
Group Contextualization for Video Recognition

```
This is official implementaion of paper "Group Contextualization for Video Recognition", which has been accepted by CVPR 2022. Paper link

```


## Overview

We release the PyTorch code of the 

## Content

- [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
- [Code](#code)
- [Pretrained Models](#pretrained-models)
  * [Kinetics-400](#kinetics-400)
    + [Dense Sample](#dense-sample)
    + [Unifrom Sampling](#unifrom-sampling)
  * [Something-Something](#something-something)
    + [Something-Something-V1](#something-something-v1)
    + [Something-Something-V2](#something-something-v2)
  * [Diving48](#Diving48)
    + [Diving48](#Diving48)
- [Testing](#testing)
- [Training](#training)
- [Live Demo on NVIDIA Jetson Nano](#live-demo-on-nvidia-jetson-nano)

## Prerequisites

The code is built with following libraries:

- [PyTorch](https://pytorch.org/) 1.0 or higher
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [tqdm](https://github.com/tqdm/tqdm.git)
- [scikit-learn](https://scikit-learn.org/stable/)

For video data pre-processing, you may need [ffmpeg](https://www.ffmpeg.org/).

## Data Preparation

We need to first extract videos into frames for fast reading. Please refer to [TSN](https://github.com/yjxiong/temporal-segment-networks) repo for the detailed guide of data pre-processing.

We have successfully trained on [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/), [UCF101](http://crcv.ucf.edu/data/UCF101.php), [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/), [Something-Something-V1](https://20bn.com/datasets/something-something/v1) and [V2](https://20bn.com/datasets/something-something/v2), [Jester](https://20bn.com/datasets/jester) datasets with this codebase. Basically, the processing of video data can be summarized into 3 steps:

- Extract frames from videos (refer to [tools/vid2img_kinetics.py](tools/vid2img_kinetics.py) for Kinetics example and [tools/vid2img_sthv2.py](tools/vid2img_sthv2.py) for Something-Something-V2 example)
- Generate annotations needed for dataloader (refer to [tools/gen_label_kinetics.py](tools/gen_label_kinetics.py) for Kinetics example, [tools/gen_label_sthv1.py](tools/gen_label_sthv1.py) for Something-Something-V1 example, and [tools/gen_label_sthv2.py](tools/gen_label_sthv2.py) for Something-Something-V2 example)
- Add the information to [ops/dataset_configs.py](ops/dataset_configs.py)

## Code

This code is based on the [TSN](https://github.com/yjxiong/temporal-segment-networks), [TSM](https://github.com/mit-han-lab/temporal-shift-module), [GST](https://github.com/chenxuluo/GST-video) and [TDN](https://github.com/MCG-NJU/TDN) codebases. 


## Pretrained Models

Here we provide some of the pretrained models. 

### Kinetics-400

#### Dense Sample

In the latest version of our paper, we reported the results of TSM trained and tested with **I3D dense sampling** (Table 1&4, 8-frame and 16-frame), using the same training and testing hyper-parameters as in [Non-local Neural Networks](https://arxiv.org/abs/1711.07971) paper to directly compare with I3D. 

We compare the I3D performance reported in Non-local paper:

| method          | n-frame      | Kinetics Acc. |
| --------------- | ------------ | ------------- |
| I3D-ResNet50    | 32 * 10clips | 73.3%         |
| TSM-ResNet50    | 8 * 10clips  | **74.1%**     |
| I3D-ResNet50 NL | 32 * 10clips | 74.9%         |
| TSM-ResNet50 NL | 8 * 10clips  | **75.6%**     |

TSM outperforms I3D under the same dense sampling protocol. NL TSM model also achieves better performance than NL I3D model. Non-local module itself improves the accuracy by 1.5%.

Here is a list of pre-trained models that we provide (see Table 3 of the paper). The accuracy is tested using full resolution setting following [here](https://github.com/facebookresearch/video-nonlocal-net). The list is keeping updating.

| model             | n-frame     | Kinetics Acc. | checkpoint                                                   | test log                                                     |
| ----------------- | ----------- | ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| TSN ResNet50 (2D) | 8 * 10clips | 70.6%         | [link](https://file.lzhu.me/projects/tsm/models/TSM_kinetics_RGB_resnet50_avg_segment5_e50.pth) | [link](https://file.lzhu.me/projects/tsm/models/log/testlog_TSM_kinetics_RGB_resnet50_avg_segment5_e50.log) |
| TSM ResNet50      | 8 * 10clips | 74.1%         | [link](https://file.lzhu.me/projects/tsm/models/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth) | [link](https://file.lzhu.me/projects/tsm/models/log/testlog_TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.log) |
| TSM ResNet50 NL   | 8 * 10clips | 75.6%         | [link](https://file.lzhu.me/projects/tsm/models/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense_nl.pth) | [link](https://file.lzhu.me/projects/tsm/models/log/testlog_TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense_nl.log) |
| TSM ResNext101    | 8 * 10clips | 76.3%         | TODO                                                         | TODO                                                         |
| TSM MoileNetV2    | 8 * 10clips | 69.5%         | [link](https://file.lzhu.me/projects/tsm/models/TSM_kinetics_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100_dense.pth) | [link](https://file.lzhu.me/projects/tsm/models/log/testlog_TSM_kinetics_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100_dense.log) |

#### Uniform Sampling

We also provide the checkpoints of TSN and TSM models using **uniform sampled frames** as in [Temporal Segment Networks](<https://arxiv.org/abs/1608.00859>) paper, which is more sample efficient and very useful for fine-tuning on other datasets. Our TSM module improves consistently over the TSN baseline.

| model             | n-frame    | acc (1-crop) | acc (10-crop) | checkpoint                                                   | test log                                                     |
| ----------------- | ---------- | ------------ | ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| TSN ResNet50 (2D) | 8 * 1clip  | 68.8%        | 69.9%         | [link](https://file.lzhu.me/projects/tsm/models/TSM_kinetics_RGB_resnet50_avg_segment5_e50.pth) | [link](https://file.lzhu.me/projects/tsm/models/log/testlog_uniform_TSM_kinetics_RGB_resnet50_avg_segment5_e50.log) |
| TSM ResNet50      | 8 * 1clip  | 71.2%        | 72.8%         | [link](https://file.lzhu.me/projects/tsm/models/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth) | [link](https://file.lzhu.me/projects/tsm/models/log/testlog_TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.log) |
| TSM ResNet50      | 16 * 1clip | 72.6%        | 73.7%         | [link](https://file.lzhu.me/projects/tsm/models/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment16_e50.pth) | -                                                            |


### Something-Something

Something-Something [V1](https://20bn.com/datasets/something-something/v1)&[V2](https://20bn.com/datasets/something-something) datasets are highly temporal-related. 

Here we provide some of the models on the dataset. The accuracy is tested using both efficient setting (center crop * 1clip) and accuate setting ([full resolution](https://github.com/facebookresearch/video-nonlocal-net) * 2clip)

#### Something-Something-V1

| model (ResNet50)  | n-frame, r | acc (full res * 2clip) | checkpoint                 | test log                                                     |
| ------------- | ------- | ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ECal-L-TSN  | 8, 16       | 48.2                   | [link](https://drive.google.com/file/d/1c8hrtADuqQA63tKCef8v7uF5v5wgf-E_/view?usp=sharing) | [link](https://drive.google.com/file/d/1hZ7-msCwcGZ2ExsZHCM2wQl2_xxToYmj/view?usp=sharing) |
| ECal-L-TSN  | 16, 16      | 48.8                   | [link](https://drive.google.com/file/d/12hR-0CX66c904lstf-eXc7k7mHrn2cAL/view?usp=sharing) | [link](https://drive.google.com/file/d/1rQW8oWf6PQ4HQdgtngGpZ9DrQXdqT3AJ/view?usp=sharing) |
| ECal-T-TSM  | 8, 16       | 49.7                   | [link](https://drive.google.com/file/d/1c8hrtADuqQA63tKCef8v7uF5v5wgf-E_/view?usp=sharing) | [link](https://drive.google.com/file/d/1N6w9W6zVCZ1sRg6A23QAeF4VVfnX16P7/view?usp=sharing) |
| ECal-T-TSM  | 16, 16      | 51.4                   | [link](https://drive.google.com/file/d/1_iPaGw5kwGmehvFob6jGgHarDiJv3wnA/view?usp=sharing) | [link](https://drive.google.com/file/d/1uqa9vzU1tJEC7ki1yXzSoUCvQwbmeQ5I/view?usp=sharing) |

#### Something-Something-V2

On V2 dataset, the accuracy is reported under the accurate setting (full resolution * 2clip).

| model (ResNet50)  | n-frame, r | acc (full res * 2clip) | checkpoint                                                   | test log                                                     |
| ------------- | --------- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ECal-L-TSN  | 8, 16       | 62.5                   | [link](https://drive.google.com/file/d/1cqa0jL9mH0TwNsz3ULAm0TRKpy76BkZ6/view?usp=sharing) | [link](https://drive.google.com/file/d/1CtA7HQ6br2UtNOlohus_8nojPAA-MUGw/view?usp=sharing) |
| ECal-L-TSN  | 16, 16      | 63.1                   | [link](https://drive.google.com/file/d/16-GbO8s0jnx2Mnt6fGjjF6i8jRy-VgtE/view?usp=sharing) | [link](https://drive.google.com/file/d/1FpoQ6_JeSMWOtUbiKGkSRD5le1wGEqYd/view?usp=sharing) |
| ECal-T-TSM  | 8, 16       | 63.5                   | [link](https://drive.google.com/file/d/1drbYjlOQT6_4OnQ5CnwMSVFEtic5U9F7/view?usp=sharing) | [link](https://drive.google.com/file/d/1KOw2neoveQehdQsuKM19I1JlbaHdbUtv/view?usp=sharing) |
| ECal-T-TSM  | 16, 16      | 65.2                   | [link](https://drive.google.com/file/d/1wXo6RRnnnsS8LoID6qtobYadepXMH28l/view?usp=sharing) | [link](https://drive.google.com/file/d/1ToiEaK-tHNGyZNL4r_HJHFiBaD-sq6rM/view?usp=sharing) |

## Testing 

For example, to test the downloaded pretrained models on Kinetics, you can run `scripts/test_tsm_kinetics_rgb_8f.sh`. The scripts will test both TSN and TSM on 8-frame setting by running:

```bash
# test TSN
python test_models.py kinetics \
    --weights=pretrained/TSM_kinetics_RGB_resnet50_avg_segment5_e50.pth \
    --test_segments=8 --test_crops=1 \
    --batch_size=64

# test TSM
python test_models.py kinetics \
    --weights=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth \
    --test_segments=8 --test_crops=1 \
    --batch_size=64
```

Change to `--test_crops=10` for 10-crop evaluation. With the above scripts, you should get around 68.8% and 71.2% results respectively.

To get the Kinetics performance of our dense sampling model under Non-local protocol, run:

```bash
# test TSN using non-local testing protocol
python test_models.py kinetics \
    --weights=pretrained/TSM_kinetics_RGB_resnet50_avg_segment5_e50.pth \
    --test_segments=8 --test_crops=3 \
    --batch_size=8 --dense_sample --full_res

# test TSM using non-local testing protocol
python test_models.py kinetics \
    --weights=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth \
    --test_segments=8 --test_crops=3 \
    --batch_size=8 --dense_sample --full_res
```

## Training 

We provided several examples to train TSM with this repo:

- To train on Kinetics from ImageNet pretrained models, you can run `scripts/train_tsm_kinetics_rgb_8f.sh`, which contains:

  ```bash
  # You should get TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth
  python main.py kinetics RGB \
       --arch resnet50 --num_segments 8 \
       --gd 20 --lr 0.02 --wd 1e-4 --lr_steps 20 40 --epochs 50 \
       --batch-size 128 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
       --shift --shift_div=8 --shift_place=blockres --npb
  ```

  You should get `TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth` as downloaded above. Notice that you should scale up the learning rate with batch size. For example, if you use a batch size of 256 you should set learning rate to 0.04.

- After getting the Kinetics pretrained models, we can fine-tune on other datasets using the Kinetics pretrained models. For example, we can fine-tune 8-frame Kinetics pre-trained model on UCF-101 dataset using **uniform sampling** by running:

  ```
  python main.py ucf101 RGB \
       --arch resnet50 --num_segments 8 \
       --gd 20 --lr 0.001 --lr_steps 10 20 --epochs 25 \
       --batch-size 64 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
       --shift --shift_div=8 --shift_place=blockres \
       --tune_from=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth
  ```

- To train on Something-Something dataset (V1&V2), using ImageNet pre-training is usually better:

  ```bash
  python main.py something RGB \
       --arch resnet50 --num_segments 8 \
       --gd 20 --lr 0.01 --lr_steps 20 40 --epochs 50 \
       --batch-size 64 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
       --shift --shift_div=8 --shift_place=blockres --npb
  ```

## Contributors
GC codes are jointly written and owned by [Dr. Yanbin Hao](https://haoyanbin918.github.io/) and [Dr. Hao Zhang](https://hzhang57.github.io/).

## Citing
```bash
@article{gc2022,
  title={Group Contextualization for Video Recognition},
  author={Yanbin Hao, Hao Zhang, Chong-Wah Ngo, Xiangnan He},
  journal={CVPR 2022},
}
```
