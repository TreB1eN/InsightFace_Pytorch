import torch.nn as nn
import pdb
import cv2
import io
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
from datetime import datetime
from PIL import Image
from torchvision import transforms as trans
from data.data_pipe import de_preprocess
from model import l2_norm
import seaborn as sns
plt.switch_backend('agg')

cosineDim1 = nn.CosineSimilarity(dim=1, eps=1e-6)


class MultipleOptimizer(object):
    def __init__(*op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()


def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'net_sphere' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn


def prepare_facebank(conf, model, mtcnn, tta=True):
    model.eval()
    embeddings = []
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


def plot_scatter(x, y, title='', xTitle='x', yTitle='y'):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel(xTitle, fontsize=14)
    plt.ylabel(yTitle, fontsize=14)
    plt.title(title, fontsize=14)
    plot = plt.scatter(x, y, linewidth=2)
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


def getTFNPString(same, isSame_pred):
    title_str = 'LL'
    if same == 1 and int(isSame_pred) == 0:
        # not the same person but predicted the same]
        # False negative
        title_str = 'False_Negative'

    elif same == 0 and int(isSame_pred) == 1:
        # False positive
        title_str = 'False_Positive'

    elif same == 1 and int(isSame_pred) == 1:
        # True positive
        title_str = 'True_Positive'

    elif same == 0 and int(isSame_pred) == 0:
        # True negative
        title_str = 'True_Negative'
    return title_str


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def heatmap_seaborn(data, row_labels, col_labels, ax=None,
            cmap=None, cbarlabel="", threshold=0.5, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        exit('no ax')
        ax = plt.gca()

    # Plot the heatmap
    g = sns.heatmap(data, ax=ax, center=threshold, vmin=-1, vmax=1,
            cmap=cmap, cbar_kws={'label': cbarlabel})

    # Create colorbar
    # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    cbar = None

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return g, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
