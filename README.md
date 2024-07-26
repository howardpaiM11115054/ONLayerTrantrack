## ONLTransTrack:On layer normalization Multiple Object Tracking with Transformer




## Introduction
 We mentioned the concept of On-layer-normalization. Changing
the structure of the Transformer in the model can effectively improve the accuracy and
optimize traniing time. We also tested the Decoder and Encoder separately and found that
On-layer normalization can be powerful under the Encoder. We will take the best results
when we set epochs to 150 and do the layer-normalization in Encoder . We achieved a
5.1% time reduction and a 3.1% improvement in IDF1.

## MOT challenge
Dataset | MOTA% | IDF1% | IDP% |IDR% |FP | FN | IDS 
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
MOT17 |  53.5 | 52.8| 65.9 |44.0 | 3102 |21017| 929



## Demo
<img src="assets/MOT17-11.gif" width="400"/>  <img src="assets/MOT17-04.gif" width="400"/>


## Installation
The codebases are built on top of [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and [CenterTrack](https://github.com/xingyizhou/CenterTrack).

#### Requirements
- Linux, CUDA>=9.2, GCC>=5.4
- Python>=3.7
- PyTorch ≥ 1.5 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this
- OpenCV is optional and needed by demo and visualization


#### Steps
1. Install and build libs
```
git clone git@github.com:howardpaiM11115054/ONLayerTrantrack.git
cd TransTrack
cd models/ops
python setup.py build install
cd ../..
pip install -r requirements.txt
```

2. Prepare datasets and annotations
```
mkdir crowdhuman
cp -r /path_to_crowdhuman_dataset/CrowdHuman_train crowdhuman/CrowdHuman_train
cp -r /path_to_crowdhuman_dataset/CrowdHuman_val crowdhuman/CrowdHuman_val
mkdir mot
cp -r /path_to_mot_dataset/train mot/train
cp -r /path_to_mot_dataset/test mot/test
```
CrowdHuman dataset is available in [CrowdHuman](https://www.crowdhuman.org/). 
```
python3 track_tools/convert_crowdhuman_to_coco.py
```
MOT dataset is available in [MOT](https://motchallenge.net/).
```
python3 track_tools/convert_mot_to_coco.py
```

3. Pre-train on crowdhuman
```
sh track_exps/crowdhuman_train.sh
python3 main_track.py  --output_dir ./output_crowdhuman --dataset_file crowdhuman --coco_path crowdhuman --batch_size 2  --with_box_refine --num_queries 500 --epochs 150 --lr_drop 100 
```


4. Train TransTrack
```
sh track_exps/crowdhuman_mot_trainhalf.sh
```

5. Evaluate TransTrack
```
sh track_exps/mot_val.sh
sh track_exps/mota.sh
```

6. Visualize TransTrack
```
python3 track_tools/txt2video.py
```





## License




## Citing

If you use TransTrack in your research or wish to refer to the baseline results published here, please use the following BibTeX entries:

```BibTeX

@

```
