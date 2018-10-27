from datetime import datetime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import io
from torchvision import transforms as trans
from data.data_pipe import de_preprocess
import torch
from model import l2_norm
import pdb
import cv2

def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn

def prepare_facebank(conf, model, mtcnn, tta = True):
    model.eval()
    embeddings =  []
    names = ['Unknown']
    for path in conf.facebank_path.iterdir():
        if path.is_file():
            continue
        else:
            embs = []
            for file in path.iterdir():
                if not file.is_file():
                    continue
                else:
                    try:
                        img = Image.open(file)
                    except:
                        continue
                    if img.size != (112, 112):
                        img = mtcnn.align(img)
                    with torch.no_grad():
                        if tta:
                            mirror = trans.functional.hflip(img)
                            emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                            emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                            embs.append(l2_norm(emb + emb_mirror))
                        else:                        
                            embs.append(model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        if len(embs) == 0:
            continue
        embedding = torch.cat(embs).mean(0,keepdim=True)
        embeddings.append(embedding)
        names.append(path.name)
    embeddings = torch.cat(embeddings)
    names = np.array(names)
    torch.save(embeddings, conf.facebank_path/'facebank.pth')
    np.save(conf.facebank_path/'names', names)
    return embeddings, names

def load_facebank(conf):
    embeddings = torch.load(conf.facebank_path/'facebank.pth')
    names = np.load(conf.facebank_path/'names.npy')
    return embeddings, names

def face_reader(conf, conn, flag, boxes_arr, result_arr, learner, mtcnn, targets, tta):
    while True:
        try:
            image = conn.recv()
        except:
            continue
        try:            
            bboxes, faces = mtcnn.align_multi(image, limit=conf.face_limit)
        except:
            bboxes = []
            
        results = learner.infer(conf, faces, targets, tta)
        
        if len(bboxes) > 0:
            print('bboxes in reader : {}'.format(bboxes))
            bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
            bboxes = bboxes.astype(int)
            bboxes = bboxes + [-1,-1,1,1] # personal choice            
            assert bboxes.shape[0] == results.shape[0],'bbox and faces number not same'
            bboxes = bboxes.reshape([-1])
            for i in range(len(boxes_arr)):
                if i < len(bboxes):
                    boxes_arr[i] = bboxes[i]
                else:
                    boxes_arr[i] = 0 
            for i in range(len(result_arr)):
                if i < len(results):
                    result_arr[i] = results[i]
                else:
                    result_arr[i] = -1 
        else:
            for i in range(len(boxes_arr)):
                boxes_arr[i] = 0 # by default,it's all 0
            for i in range(len(result_arr)):
                result_arr[i] = -1 # by default,it's all -1
        print('boxes_arr ： {}'.format(boxes_arr[:4]))
        print('result_arr ： {}'.format(result_arr[:4]))
        flag.value = 0

hflip = trans.Compose([
            de_preprocess,
            trans.ToPILImage(),
            trans.functional.hflip,
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)
    return hfliped_imgs

def get_time():
    return (str(datetime.now())[:-10]).replace(' ','-').replace(':','-')

def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plot = plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return buf

def draw_box_name(bbox,name,frame):
    frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),6)
    frame = cv2.putText(frame,
                    name,
                    (bbox[0],bbox[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    2,
                    (0,255,0),
                    3,
                    cv2.LINE_AA)
    return frame