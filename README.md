# Group Contextualization for Video Recognition
This is official implementaion of paper "Group Contextualization for Video Recognition", which has been accepted by CVPR 2022. [`Paper link`]()
<div align="center">
  <img src="demo/GC.PNG" width="800px"/>
</div>


## Updates
### March 11, 2022
* Release this V1 version (the version used in paper) to public.

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
* PyTorch >= 1.7, torchvision
* tensorboardx

For video data pre-processing, you may need [ffmpeg](https://www.ffmpeg.org/).

## Data Preparation

For GC-TSN, GC-GST, GC-TSM, we need to first extract videos into frames for all datasets ([Kinetics-400](https://deepmind.com/research/open-source/open-source-datasets/kinetics/), [Something-Something V1](https://20bn.com/datasets/something-something/v1) and [V2](https://20bn.com/datasets/something-something/v2), [Diving48](http://www.svcl.ucsd.edu/projects/resound/dataset.html) and [EGTEA Gaze+](http://cbi.gatech.edu/fpv)), following the [TSN](https://github.com/yjxiong/temporal-segment-networks) repo. While for GC-TDN, the data process follows the backbone [TDN](https://github.com/MCG-NJU/TDN) work, which resizes the short edge of video to 320px and directly decodes video mp4 file during training/evaluation.


## Code

GC-TSN/TSM/GST/TDN codes are based on [TSN](https://github.com/yjxiong/temporal-segment-networks), [TSM](https://github.com/mit-han-lab/temporal-shift-module), [GST](https://github.com/chenxuluo/GST-video) and [TDN](https://github.com/MCG-NJU/TDN) codebases, respectively. 


## Pretrained Models

Here we provide some of the pretrained models. 

### Kinetics-400

| Model             | Frame * view * clip    | Top-1 Acc. | Top-5 Acc. | Checkpoint |
| ----------------- | ----------- | ---------- | ----------- | ---------------- |
| GC-TSN ResNet50   | 8 * 1 * 10  | 75.2%      | 92.1%     | [link]() |
| GC-TSM ResNet50   | 8 * 1 * 10  | 75.4%      | 91.9%     | [link]() |
| GC-TSM ResNet50   | 16 * 1 * 10 | 76.7%      | 92.9%     | [link]() |
| GC-TSM ResNet50   | 16 * 3 * 10 | 77.1%      | 92.9%     | [link]() |
| GC-TDN ResNet50   | 8 * 3 * 10  | 77.3%      | 93.2%     | [link]() |
| GC-TDN ResNet50   | 16 * 3 * 10  | 78.8%      | 93.8%     | [link]() |
| GC-TDN ResNet50   | (8+16) * 3 * 10  | 79.6%   | 94.1%     | [link]() |


### Something-Something

Something-Something [V1](https://20bn.com/datasets/something-something/v1)&[V2](https://20bn.com/datasets/something-something) datasets are highly temporal-related. Here, we 
use the 224×224 center crop for performance report.

#### Something-Something-V1

| Model             | Frame * view * clip    | Top-1 Acc. | Top-5 Acc. | Checkpoint |
| ----------------- | ----------- | ---------- | ----------- | ---------------- |
| GC-TSN ResNet50   | 8 * 1 * 2  | 75.2%      | 92.1%     | [link]() |
| GC-TSN ResNet50   | 16 * 1 * 2  | 75.2%      | 92.1%     | [link]() |
| GC-TSN ResNet50   | (8+16) * 1 * 2  | 75.2%      | 92.1%     | [link]() |
| GC-TSM ResNet50   | 8 * 1 * 10  | 75.4%      | 91.9%     | [link]() |
| GC-TSM ResNet50   | 16 * 1 * 10 | 76.7%      | 92.9%     | [link]() |
| GC-TSM ResNet50   | 16 * 3 * 10 | 77.1%      | 92.9%     | [link]() |
| GC-TDN ResNet50   | 8 * 3 * 10  | 77.3%      | 93.2%     | [link]() |
| GC-TDN ResNet50   | 16 * 3 * 10  | 78.8%      | 93.8%     | [link]() |
| GC-TDN ResNet50   | (8+16) * 3 * 10  | 79.6%   | 94.1%     | [link]() |

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
