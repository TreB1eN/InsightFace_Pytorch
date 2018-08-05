# InsightFace_Pytorch
Pytorch0.4 codes for InsightFace

- - -
## 1.Intro
* This repo is a reimplementation of Arcface[(paper)](https://arxiv.org/abs/1801.07698), or Insightface[(github)](https://github.com/deepinsight/insightface)
* For models, including the pytorch implementation of the backbone modules of Arcface and MobileFacenet
* Codes for transform MXNET data records in Insightface[(github)](https://github.com/deepinsight/insightface) to Image Datafolders are provided
* Pretrained models are posted, include the [MobileFacenet](https://arxiv.org/abs/1804.07573) and IR-SE50 in the original paper
- - -
## 2. Pretrained Models & Performance
[MobileFaceNet@BaiduDrive](Coming Soon), [@GoogleDrive](Coming Soon) (coming soon)

|  LFW(%) | CFP-FF(%) | CFP-FP(%) | AgeDB-30(%) | calfw(%) | cplfw(%) | calfw(%) | vgg2_fp(%)   |
|  ------     | --------- | --------- | ----------- | ------------- |------------- |------------- |
|   0.9952      | 0.9962     | 0.9504     | 0.9622      |    0.9557     |    0.9107     |    0.9386     |  

[LResNet50E-IR@BaiduDrive](Coming Soon), [@GoogleDrive](Coming Soon) (coming soon)

|  LFW(%)     | CFP-FF(%) | CFP-FP(%) | AgeDB-30(%) | MegaFace(%)   |
|  ------     | --------- | --------- | ----------- | ------------- |
|   0.9952      | 0.9962     | 0.9504     | 0.9622       | ?        |

## 3.How to use
* clone
    ```
    git clone https://github.com/TropComplique/mtcnn-pytorch.git
    ```
### 3.1Data Preparation
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
#### 3.2.2 Prepare Dataset ( For training)
download the refined dataset from original post: (emore recommended)
* [Refined-MS1M@BaiduDrive](https://pan.baidu.com/s/1nxmSCch), [Refined-MS1M@GoogleDrive](https://drive.google.com/file/d/1XRdCt3xOw7B3saw0xUSzLRub_HI4Jbk3/view)
* [VGGFace2@BaiduDrive](https://pan.baidu.com/s/1c3KeLzy), [VGGFace2@GoogleDrive](https://drive.google.com/open?id=1KORwx_DWyIScAjD6vbo4CSRu048APoum)
* [emore dataset @ BaiduDrive](https://pan.baidu.com/s/1c3KeLzy), [emore dataset @ OneDrive](https://pan.baidu.com/s/1c3KeLzy)

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
### detect over camera:

* 2 download the desired weights [Mobilefacenet] , [IR-SE50]to model folder
[Mobilefacenet]:https://pan.baidu.com/s/1PwHjtGLAmAoG5LJkQk5LSQ
[IR-SE50]:https://pan.baidu.com/s/1PwHjtGLAmAoG5LJkQk5LSQ

* 3 to take a picture, run
    ```
    python take_pic.py -n name
    ```
    press q to take a picture, it will only capture 1 highest possibility face if more than 1 person appear in the camera

* 4 or you can put any preexisting photo into the facebank directory, the file structure is following:
    
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

- 5 to start
    ```
    python face_verify.py 
    ```
- - -
### detect over video:
    ```
    python infer_on_video.py -f [video file name] -s [save file name]
    ```
the video file should be inside the data/face_bank folder

- Video Detection Demo [@Youtube](https://www.youtube.com/watch?v=6r9RCRmxtHE)

### Training:
    ```
    python train.py -b [batch_size] -lr [learning rate] -e [epochs]
    ```
