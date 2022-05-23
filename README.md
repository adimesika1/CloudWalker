# CloudWalker: Random walks for 3D point cloud shape analysis
![Alt text](images/teaser.PNG?raw=true "Title")

## [[Paper]](https://arxiv.org/abs/2112.01050)
Created by [Adi Mesika](mailto:adimesika10@gmail.com) from Technion - Israel Institute of Technology

This repository contains the implementation of CloudWalker

## Installation
The code is tested under tf-2.4.1 GPU version and python 3.8 on Ubunto 18.04, Cuda 11.1, GPU RTX 3090.
There are also some dependencies for a few Python libraries for data processing and visualizations (requirements file).


### Raw datasets
To get the raw datasets go to the relevant website, 
and put it under `CloudWalker/datasets_raw/<dataset>`. 
- [ModelNet](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip)
- [3DFuture](https://tianchi.aliyun.com/dataset/dataDetail?spm=5176.14208604.0.0.53c83cf7kHDv5j&dataId=98063) (3D-FUTURE-model.zip).
- [ScanObjectNN](https://hkust-vgd.github.io/scanobjectnn/) (fill out an agreement).


### Processed
To prepare the data, run `python dataset_prepare.py <dataset>`

Processing will rearrange dataset in `npz` files.
Then, copy the generated directory to `CloudWalker/datasets_processed/<dataset>`.

The model can be run in two ways. On one, walks that are already prepared, and on the other, walks that are created during training. We decided to release the most efficient version.
It currently requires the walks to be pre-created.

To prepare the walks, run `python pre_created_walks/save_walk_as_npz.py <dataset>`


## Training
```
python train_val.py <dataset>
```
While `<dataset>` can be one of the following: 
`modelnet40_normal_resampled` / `3dfuture` / `scanobjectnn`.

You will find the results at: `CloudWalker\runs\`

To get the final accuracy results, please refer to the "log.txt" file at `CloudWalker\runs\<trained_model>`, 
or run evaluation script.

## Evaluating
After training is finished (or pretrained is downloaded),
To evaluate **classification** task run: 
```
python evaluate_classification.py <dataset> <trained_model_directory>
```

## Citation
If you find our work useful in your research, please consider citing:
```
@article{mesika2021cloudwalker,
  title={CloudWalker: 3D Point Cloud Learning by Random Walks for Shape Analysis},
  author={Mesika, Adi and Ben-Shabat, Yizhak and Tal, Ayellet},
  journal={arXiv preprint arXiv:2112.01050},
  year={2021}
}
```

## Questions / Issues
If you have questions or issues running this code, please open an issue.
