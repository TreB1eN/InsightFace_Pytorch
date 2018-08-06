# InsightFace_Pytorch
Pytorch0.4 codes for InsightFace

- - -
## 1. Intro
* This repo is a reimplementation of Arcface[(paper)](https://arxiv.org/abs/1801.07698), or Insightface[(github)](https://github.com/deepinsight/insightface)
* For models, including the pytorch implementation of the backbone modules of Arcface and MobileFacenet
* Codes for transform MXNET data records in Insightface[(github)](https://github.com/deepinsight/insightface) to Image Datafolders are provided
* Pretrained models are posted, include the [MobileFacenet](https://arxiv.org/abs/1804.07573) and IR-SE50 in the original paper
- - -
## 2. Pretrained Models & Performance
[MobileFaceNet@BaiduDrive](Coming Soon), [@GoogleDrive](Coming Soon) (coming soon)

|  LFW(%) | CFP-FF(%) | CFP-FP(%) | AgeDB-30(%) | calfw(%) | cplfw(%) |  vgg2_fp(%)   |
|  ------     | --------- | --------- | ----------- | ------------- | ------------- | ------------- |
|   0.9952      | 0.9962     | 0.9504     | 0.9622      |    0.9557     |    0.9107     |    0.9386        |

[LResNet50E-IR@BaiduDrive](Coming Soon), [@GoogleDrive](Coming Soon) (coming soon)

|  LFW(%) | CFP-FF(%) | CFP-FP(%) | AgeDB-30(%) | calfw(%) | cplfw(%) |  vgg2_fp(%)   |
|  ------     | --------- | --------- | ----------- | ------------- | ------------- | ------------- |
|   ?      | ?     | ?     | ?      |    ?     |    ?     |    ?        |

## 3. How to use
* clone
    ```
    git clone https://github.com/TropComplique/mtcnn-pytorch.git
    ```
### 3.1 Data Preparation
#### 3.1.1 Prepare Facebank (For testing over camera or video)
Provide the face images your want to detect in the data/face_bank folder, and guarantee it have a structure like following:
```
data/facebank/
        ---> id1/
            ---> id1_1.jpg
        ---> id2/
            ---> id2_1.jpg
        ---> id3/
            ---> id3_1.jpg
           ---> id3_2.jpg
```
#### 3.1.2 download the pretrained model to work_space/model
If more than 1 image appears in one folder, an average embedding will be calculated
#### 3.2.3 Prepare Dataset ( For training)
download the refined dataset from original post: (emore recommended)
* [Refined-MS1M@BaiduDrive](https://pan.baidu.com/s/1nxmSCch), [Refined-MS1M@GoogleDrive](https://drive.google.com/file/d/1XRdCt3xOw7B3saw0xUSzLRub_HI4Jbk3/view)
* [VGGFace2@BaiduDrive](https://pan.baidu.com/s/1c3KeLzy), [VGGFace2@GoogleDrive](https://drive.google.com/open?id=1KORwx_DWyIScAjD6vbo4CSRu048APoum)
* [emore dataset @ BaiduDrive](https://pan.baidu.com/s/1c3KeLzy), [emore dataset @ OneDrive](https://pan.baidu.com/s/1c3KeLzy)

**Note:** If you use the refined [MS1M](https://arxiv.org/abs/1607.08221) dataset and the cropped [VGG2](https://arxiv.org/abs/1710.08092) dataset, please cite the original papers.

* after unzip the files to 'data' path, run :
    ```
    python prepare_data.py
    ```
after the execution, you should find following structure:
```
faces_emore/
            ---> agedb_30
            ---> calfw
            ---> cfp_ff
            --->  cfp_fp
            ---> cfp_fp
            ---> cplfw
            --->imgs
            ---> lfw
            ---> vgg2_fp
```
- - -
### 3.2 detect over camera:

* 1. download the desired weights to model folder:
- [IR-SE50 @ BaiduNetdisk](https://pan.baidu.com/s/12BUjjwy1uUTEF9HCx5qvoQ)
- [IR-SE50 @ Onedrive](https://1drv.ms/u/s!AhMqVPD44cDOhkPsOU2S_HFpY9dC)
- [Mobilefacenet @ BaiduNetDisk](https://pan.baidu.com/s/1hqNNkcAjQOSxUjofboN6qg)
- [Mobilefacenet @ OneDrive (comming soon)]()

* 2 to take a picture, run
    ```
    python take_pic.py -n name
    ```
    press q to take a picture, it will only capture 1 highest possibility face if more than 1 person appear in the camera

* 3 or you can put any preexisting photo into the facebank directory, the file structure is following:
    
- facebank/
         name1/
             photo1.jpg
             photo2.jpg
             ...
         name2/
             photo1.jpg
             photo2.jpg
             ...
         .....
    if more than 1 image appears in the directory, average embedding will be calculated

- 4 to start
    ```
    python face_verify.py 
    ```
- - -
### 3.3 detect over video:
    ```
    python infer_on_video.py -f [video file name] -s [save file name]
    ```
the video file should be inside the data/face_bank folder

- Video Detection Demo [@Youtube](https://www.youtube.com/watch?v=6r9RCRmxtHE)

### 3.4 Training:
    ```
    python train.py -b [batch_size] -lr [learning rate] -e [epochs]
    ```
## 4. References 
* This repo is mainly inspired by [deepinsight/insightface](https://github.com/deepinsight/insightface) and [InsightFace_TF](https://github.com/auroua/InsightFace_TF)

## PS
* PRs are welcome, in case that I don't have the resource to train some large models like the 100 and 151 layers model
* Email : treb1en@qq.com
