from easydict import EasyDict as edict
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans

def get_config(training = True):
    conf = edict()
    conf.fixed_str = 'ir_se50.pth'
    conf.pretrainedMdl = 'ir_se50.pth'
    # conf.fixed_str = '2019-08-24-10-31_accuracy:0.9194285714285716_step:76660_None.pth'
    conf.data_path = Path('data')
    conf.work_path = Path('work_space/')
    conf.model_path = conf.work_path/'models'
    conf.log_path = conf.work_path/'log/'
    conf.save_path = conf.work_path/'save'
    conf.exp_title = 'xCos'
    conf.exp_comment = 'expMS1M'
    conf.input_size = [112,112]
    conf.embedding_size = 1568 #3136 #512
    conf.use_mobilfacenet = False
    conf.net_depth = 50
    conf.drop_ratio = 0.6
    conf.net_mode = 'ir_se' # or 'ir'
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.test_transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    conf.data_mode = 'emore'
    conf.vgg_folder = conf.data_path/'faces_vgg_112x112'
    conf.ms1m_folder = conf.data_path/'faces_ms1m_112x112'
    conf.emore_folder = conf.data_path/'faces_emore'#'faces_webface_112x112' # 'faces_emore'
    conf.batch_size = 100 # irse net depth 50
#   conf.batch_size = 200 # mobilefacenet
#--------------------Training Config ------------------------
    conf.USE_SOFTMAX = True
    conf.SOFTMAX_T = 1
    if training:
        conf.log_path = conf.work_path/'log'
        conf.save_path = conf.work_path/'save'
    #     conf.weight_decay = 5e-4
        conf.lr = 1e-3
        conf.milestones = [12,15,18]
        conf.momentum = 0.9
        conf.pin_memory = True
#         conf.num_workers = 4 # when batchsize is 200
        conf.num_workers = 1
        conf.ce_loss = CrossEntropyLoss()
#--------------------Inference Config ------------------------
    else:
        conf.facebank_path = conf.data_path/'facebank'
        conf.threshold = 1.5
        conf.threshold_xCos = 0.2338
        conf.face_limit = 10
        #when inference, at maximum detect 10 faces in one image, my laptop is slow
        conf.min_face_size = 30
        # the larger this value, the faster deduction, comes with tradeoff in small faces
    return conf
