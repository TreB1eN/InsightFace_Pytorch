import os
import math
import cv2
import bcolz
import torch
import numpy as np
from torch import optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms as trans

from utils import get_time, gen_plot, plot_scatter, hflip_batch, \
        separate_bn_paras, cosineDim1, MultipleOptimizer,\
        getTFNPString, heatmap, heatmap_seaborn, annotate_heatmap
from data.data_pipe import de_preprocess, get_train_loader, get_val_data, loader_from_carray
from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm, Backbone_FC2Conv
from networks import AttentionXCosNet
from net_sphere import sphere20a, AngleLoss, AngleLinear
from verification import evaluate, evaluate_attention
from losses import l2normalize, CosAttentionLoss
plt.switch_backend('agg')


class face_learner(object):
    def __init__(self, conf, inference=False):
        print(conf)
        self.conf = conf
        if conf.use_mobilfacenet:
            self.model = MobileFaceNet(conf.embedding_size).to(conf.device)
            print('MobileFaceNet model generated')
        elif conf.modelType == 'ArcFace':
            self.model = Backbone_FC2Conv(conf.net_depth,
                                          conf.drop_ratio,
                                          conf.net_mode).to(conf.device)
            print('{}_{} model generated'.format(conf.net_mode,
                                                 conf.net_depth))
        elif conf.modelType == 'CosFace':
            self.model = Backbone_FC2Conv(conf.net_depth,
                                          conf.drop_ratio,
                                          conf.net_mode).to(conf.device)
            print('{}_{} model generated'.format(conf.net_mode,
                                                 conf.net_depth))
        elif conf.modelType == 'SphereFace':
            self.model = sphere20a(returnGrid=True).to(conf.device)
            print('>>> SphereFace model is generated.')

        # Attention Model
        self.model_attention = AttentionXCosNet(conf).to(conf.device)
        self.xCos_loss_with_attention = CosAttentionLoss()
        # Create model_gt for cos_gt generation
        self.model_tgt = Backbone(conf.net_depth,
                                  conf.drop_ratio, conf.net_mode)
        self.model_tgt = self.model_tgt.to(conf.device)
        self.model_tgt.load_state_dict(torch.load(conf.save_path/'model_{}'
                                       .format(conf.pretrainedMdl)))
        self.model_tgt = self.model_tgt.eval()

        if inference:
            self.threshold = conf.threshold
            self.threshold_xCos = conf.threshold_xCos
        else:  # Training mode

            self.milestones = conf.milestones

            self.loader, self.class_num = get_train_loader(conf)
            if conf.modelType == 'ArcFace':
                self.head = Arcface(embedding_size=conf.embedding_size,
                                    classnum=self.class_num).to(conf.device)
                self.loss_fr = CrossEntropyLoss()
            elif conf.modelType == 'CosFace':
                self.head = Am_softmax(embedding_size=conf.embedding_size,
                                       classnum=self.class_num).to(conf.device)
                self.loss_fr = CrossEntropyLoss()
            elif conf.modelType == 'SphereFace':
                self.head = AngleLinear(conf.embedding_size,
                                        self.class_num).to(conf.device)
                self.loss_fr = AngleLoss()

            self.writer = SummaryWriter(os.path.join(conf.log_path,
                                        conf.exp_title + '/' +
                                        conf.exp_comment))
            self.step = 0

            print('two model heads generated')

            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)

            if conf.use_mobilfacenet:
                self.optimizer_fr = optim.SGD([
                                    {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                                    {'params': [paras_wo_bn[-1]] + [self.head.kernel], 'weight_decay': 4e-4},
                                    {'params': paras_only_bn}
                                ], lr = conf.lr, momentum = conf.momentum)
            else:
                self.optimizer_fr = optim.SGD([
                                    {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-4},
                                    {'params': paras_only_bn}
                                ], lr = conf.lr, momentum = conf.momentum)
                print(self.optimizer_fr)
            #TODO 0827
            self.optimizer_atten = optim.Adam(self.model_attention.parameters(), lr=conf.lr)
            # self.optimizer = MultipleOptimizer(self.optimizer_fr, self.optimizer_atten
#             self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=40, verbose=True)
            self.scheduler_atten = optim.lr_scheduler.StepLR(self.optimizer_atten, 8, gamma=0.99, last_epoch=-1)
            print('optimizers generated')
            self.board_loss_every = len(self.loader) // 100
            self.evaluate_every = len(self.loader) // 10#10
            self.save_every = len(self.loader)//5
            '''
            self.lfw: (12000, 3, 112, 112)
            self.lfw_issame: (6000,)
            '''
            self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame = get_val_data(self.loader.dataset.root.parent)


    def getCos(self, imgs):
        '''
        imgs.size: [bs * 2, c, h, w]
        feats: [bs * 2, 512]
        feat1: [bs, 512]
        cosine:(bs,)
        '''
        with torch.no_grad():
            feats = self.model_tgt(imgs)
            half_idx = feats.size(0) // 2
            feat1 = feats[:half_idx]
            feat2 = feats[half_idx:]
            feat1 = l2normalize(feat1)
            feat2 = l2normalize(feat2)
            cosine = cosineDim1(feat1, feat2)
            return cosine

    def getCosFrom2Imgs(self, img1s, img2s):
        '''
        img1s: tensor of size (conf.bs//2, 3, 112, 112)
        img2s: tensor of size (conf.bs//2, 3, 112, 112)
        feat1: tensor of size [conf.bs//2, 512]
        feat2: tensor of size [conf.bs//2, 512]
        cosine:tensor of size (bs//2,)
        '''
        with torch.no_grad():
            feat1 = self.model_tgt(img1s)
            feat2 = self.model_tgt(img2s)
            feat1 = l2normalize(feat1)
            feat2 = l2normalize(feat2)
            cosine = cosineDim1(feat1, feat2)
            return cosine

    def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        time_log = get_time()
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        torch.save(
            self.model.state_dict(), save_path /
            ('model_{}_accuracy:{}_step:{}_{}.pth'.format(time_log, accuracy, self.step, extra)))
        torch.save(
            self.model_attention.state_dict(), save_path /
            ('model_attention_{}_accuracy:{}_step:{}_{}.pth'.format(time_log, accuracy, self.step, extra)))
        if not model_only:
            torch.save(
                self.head.state_dict(), save_path /
                ('head_{}_accuracy:{}_step:{}_{}.pth'.format(time_log, accuracy, self.step, extra)))
            torch.save(
                self.optimizer_fr.state_dict(), save_path /
                ('optimizer_fr_{}_accuracy:{}_step:{}_{}.pth'.format(time_log, accuracy, self.step, extra)))
            torch.save(
                self.optimizer_atten.state_dict(), save_path /
                ('optimizer_atten_{}_accuracy:{}_step:{}_{}.pth'.format(time_log, accuracy, self.step, extra)))

    def load_state(self, conf, fixed_str,
                   from_save_folder=False,
                   model_only=False,
                   model_atten=True,
                   strict=True):
        if from_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        if conf.modelType == 'SphereFace':
            print('>>>> Loading Sphereface weights')
            self.model.load_state_dict(torch.load(save_path/'sphere20a_20171020.pth'), strict=strict)
        else:
            self.model.load_state_dict(torch.load(save_path/'model_{}'.format(fixed_str)), strict=strict)
        if model_atten:
            self.model_attention.load_state_dict(torch.load(save_path/'model_attention_{}'.format(fixed_str)), strict=strict)
        if not model_only:
            self.head.load_state_dict(torch.load(save_path/'head_{}'.format(fixed_str)))
            self.optimizer_fr.load_state_dict(torch.load(save_path/'optimizer_fr_{}'.format(fixed_str)))
            self.optimizer_atten.load_state_dict(torch.load(save_path/'optimizer_atten_{}'.format(fixed_str)))

    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor):
        self.writer.add_scalar(self.conf.exp_title + '/{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_scalar(self.conf.exp_title + '/{}_best_threshold'.format(db_name), best_threshold, self.step)
        self.writer.add_image(self.conf.exp_title + '/{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)
#         self.writer.add_scalar('{}_val:true accept ratio'.format(db_name), val, self.step)
#         self.writer.add_scalar('{}_val_std'.format(db_name), val_std, self.step)
#         self.writer.add_scalar('{}_far:False Acceptance Ratio'.format(db_name), far, self.step)

    def evaluate(self, conf, carray, issame, nrof_folds = 5, tta = False):
        '''
        carray: list (2 * # of pairs, 3, 112, 112)
        issame: list (# of pairs,)
        '''
        self.model.eval()
        self.model.returnGrid = False  # Remember to reset this before return!
        idx = 0
        embeddings = np.zeros([len(carray), conf.embedding_size])
        with torch.no_grad():
            while idx + conf.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + conf.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch)
                else:
                    embeddings[idx:idx + conf.batch_size] = self.model(batch.to(conf.device)).cpu()
                idx += conf.batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    embeddings[idx:] = self.model(batch.to(conf.device)).cpu()
        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        self.model.returnGrid = True
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor

    def _to_loader(self, carray_or_loader, conf):
        if isinstance(carray_or_loader, torch.utils.data.DataLoader):
            loader = carray_or_loader
        elif isinstance(carray_or_loader, bcolz.carray_ext.carray):
            loader = loader_from_carray(carray_or_loader, conf)
        else:
            raise NotImplementedError()
        return loader

    def getXCos(self, carray, conf, tta=False, attention=None, returnCosGt=False, returnXCAP=False):
        '''
        returnXCAP: return xCoses, Coses, attentionMaps, cosPatchedMaps
        '''
        self.model.eval()
        self.model.returnGrid = True  # Remember to reset this before return!
        self.model_attention.eval()
        loader = self._to_loader(carray, conf)

        output_keys = ['xCos', 'gtCos', 'cosPatchedMap', 'attentionMap']
        output_dict = {key: [] for key in output_keys}

        with torch.no_grad():
            def batch2feat(batch, conf, tta):
                if tta:
                    fliped = hflip_batch(batch)
                    femb_bat1, emb_batch_1 = self.model(batch.to(conf.device))
                    femb_bat2, emb_batch_2 = self.model(fliped.to(conf.device))
                    # XXX or l2norm?
                    emb_batch = (emb_batch_1 + emb_batch_2) / 2
                    femb_bat = (femb_bat1 + femb_bat2) / 2
                else:
                    femb_bat, emb_batch = self.model(batch.to(conf.device))
                return femb_bat, emb_batch

            def computeXCosWithAttention(emb_batch, attention):
                # Calculate Attention
                grid_feat1 = emb_batch[0::2]  # grid_feat1: (bs//2, 32, 7, 7)
                grid_feat2 = emb_batch[1::2]
                if attention is None:
                    # Size of attention: (bs//2, 1, 7, 7)
                    attention = self.model_attention(grid_feat1, grid_feat2)
                # Size of xCos: (bs//2,)
                xCos, cos_patched = self.xCos_loss_with_attention.computeXCos(
                        grid_feat1, grid_feat2, attention,
                        returnCosPatched=True)
                # Squeeze for channel dimension
                attention = torch.squeeze(attention.permute(0, 2, 3, 1))
                return xCos, attention, cos_patched

            def batch2XCosAndGtCos(batch, attention, conf, tta):
                '''
                batch: tensor of size (bs, 3, 112, 112) A[0::2] B[1::2]
                '''
                femb_bat, emb_batch = batch2feat(batch, conf, tta)

                xCos, attentionMap, cos_patched = computeXCosWithAttention(
                        emb_batch, attention)
                # XXX
                # gtCos = cosineDim1(femb_bat[0::2], femb_bat[1::2])
                gtCos = self.getCosFrom2Imgs(batch[0::2].to(conf.device),
                                             batch[1::2].to(conf.device))
                # Store batch to xCos matrix
                xCos = xCos.cpu().numpy()
                gtCos = gtCos.cpu().numpy()
                cos_patched = cos_patched.cpu().numpy()
                attentionMap = attentionMap.cpu().numpy()
                return xCos, gtCos, cos_patched, attentionMap

            for batch in loader:
                xCos, gtCos, cos_patched, attentionMap = batch2XCosAndGtCos(batch, attention, conf, tta)
                for output_key, output in zip(output_keys, [xCos, gtCos, cos_patched, attentionMap]):
                    output_dict[output_key].append(output)

            for key in output_keys:
                output_dict[key] = np.concatenate(output_dict[key], axis=0)

        if returnXCAP:
            return (output_dict[key] for key in output_keys)

        if returnCosGt:
            return (output_dict[key] for key in ['xCos', 'gtCos'])
        else:
            return output_dict['xCos']

    def get_original_cosines(self, carray, conf, tta=False):
        self.model.eval()
        self.model.returnGrid = True  # Remember to reset this before return!
        self.model_attention.eval()
        assert conf.batch_size % 2 == 0, "Need even batch size"
        loader = self._to_loader(carray, conf)

        cosines = []
        with torch.no_grad():
            for batch in loader:
                if tta:
                    fliped = hflip_batch(batch)
                    feat_orig = self.model.get_original_feature(batch.to(conf.device))
                    feat_flip = self.model.get_original_feature(fliped.to(conf.device))
                    feat = (feat_orig + feat_flip) / 2
                else:
                    feat = self.model.get_original_feature(batch)
                feat_left_person = l2normalize(feat[0::2])
                feat_right_person = l2normalize(feat[1::2])
                cosine = cosineDim1(feat_left_person, feat_right_person)
                cosines.append(cosine.cpu().numpy())
        return np.concatenate(cosines, axis=0)

    def evaluate_attention(self, conf, carray, issame,
                           nrof_folds=5, tta=False, attention=None):
        '''
        carray: list (2 * # of pairs, 3, 112, 112)
        issame: list (# of pairs,)
        emb_batch: tensor [bs, 32, 7, 7]
        xCoses: list (# of pairs,)
        attention: GPUtorch.FloatTensor((bs//2, 1, 7, 7)),is ones/sum() or corr
        '''
        xCoses = self.getXCos(carray, conf, tta=tta, attention=attention)
        return self.evaluate_and_plot_roc(xCoses, issame, nrof_folds)

    def evaluate_and_plot_roc(self, coses, issame, nrof_folds=5):
        tpr, fpr, accuracy, best_thresholds = evaluate_attention(coses, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor

    def plot_CorrBtwXCosAndCos(self, conf, carray, issame,
                               nrof_folds=5, tta=False, attention=None):
        '''
        carray: list (2 * # of pairs, 3, 112, 112)
        issame: list (# of pairs,)
        emb_batch: tensor [bs, 32, 7, 7]
        xCoses: list (# of pairs,)
        attention: GPUtorch.FloatTensor((bs//2, 1, 7, 7)),is ones/sum() or corr
        '''
        xCoses, gtCoses = self.getXCos(carray, conf, tta=tta, attention=attention, returnCosGt=True)
        # tpr, fpr, accuracy, best_thresholds = evaluate_attention(
        #         xCoses, issame, nrof_folds)
        title = 'xCos vs Cos on lfw 6000 pairs'
        buf = plot_scatter(xCoses, gtCoses, title, 'xCos', 'Cos')
        corrPlot = Image.open(buf)
        corrPlot_tensor = trans.ToTensor()(corrPlot)
        return corrPlot_tensor

    def plot_CorrBtwPatchCosAndGtCos(self, conf, carray, issame,
                                     nrof_folds=5, tta=False, attention=None):
        '''
        carray: list (2 * # of pairs, 3, 112, 112)
        issame: list (# of pairs,)
        emb_batch: tensor [bs, 32, 7, 7]
        xCoses: list (# of pairs,)
        attention: GPUtorch.FloatTensor((bs//2, 1, 7, 7)),is ones/sum() or corr
        '''
        def getCorr(s_matrix, vec):
            _, h, w = s_matrix.shape
            result = np.zeros((h, w))
            for i in range(h):
                for j in range(w):
                    result[i][j] = np.corrcoef(s_matrix[:, i, j], vec)[0, 1]
            return result
        xCoses, gtCoses, attentionMaps, cosPatchedMaps = self.getXCos(
                carray, conf, tta=tta, attention=attention, returnXCAP=True)
        # Calculate the correlation
        corr_result = getCorr(cosPatchedMaps, gtCoses)
        # plt.matshow(corr_result)
        fig, ax = plt.subplots()

        im, cbar = heatmap(corr_result, [], [], ax=ax,
                                   cmap="YlGn", cbarlabel="correlation with GAP feat")
        texts = annotate_heatmap(im, valfmt="{x:.3f}")

        fig.tight_layout()
        plt.show()
        plt.gcf().clear()

        title = 'xCos vs Cos on lfw 6000 pairs'
        buf = plot_scatter(xCoses, gtCoses, title, 'xCos', 'Cos')
        corrPlot = Image.open(buf)
        corrPlot_tensor = trans.ToTensor()(corrPlot)
        return corrPlot_tensor, corr_result

    def plot_Examples(self, conf, carray, issame,
                      nrof_folds=5, tta=False, attention=None,
                      exDir='defaultExamples'):
        '''
        carray: list (2 * # of pairs, 3, 112, 112)
        issame: list (# of pairs,)
        emb_batch: tensor [bs, 32, 7, 7]
        xCoses: list (# of pairs,)
        attention: GPUtorch.FloatTensor((bs//2, 1, 7, 7)),is ones/sum() or corr
        '''
        exPath = str(self.conf.work_path) + '/' + exDir
        if not os.path.exists(exPath):
            os.makedirs(exPath)

        xCoses, gtCoses, attentionMaps, cosPatchedMaps = self.getXCos(
                carray, conf, tta=tta, attention=attention, returnXCAP=True)
        # tpr, fpr, accuracy, best_thresholds = evaluate_attention(
        #         xCoses, issame, nrof_folds)

        threshold = conf.threshold_xCos
        for i, xCos in enumerate(tqdm(xCoses)):
            gtCos = gtCoses[i]
            attentionMap = attentionMaps[i]
            cosPatchedMap = cosPatchedMaps[i]
            img1Idx = i * 2
            img2Idx = img1Idx + 1
            img1 = ((carray[img1Idx] * 0.5 + 0.5) * 255).astype('uint8')
            img2 = ((carray[img2Idx] * 0.5 + 0.5) * 255).astype('uint8')
            isTheSamePerson = issame[i]

            self.plot_attention_example(gtCos, xCos, threshold,
                    cosPatchedMap, attentionMap, img1, img2,
                    isTheSamePerson, exPath)
        #TODO
        # buf = plot_scatter(xCoses, gtCoses, title, 'xCos', 'Cos')
        # corrPlot = Image.open(buf)
        # corrPlot_tensor = trans.ToTensor()(corrPlot)
        return

    def plot_attention_example(self, cos_fr, cos_x, threshold,
                               cos_patch, weight_attention,
                               image1, image2, isSame, exPath):
        # XXX This function can be moved to utils.py?
        name1, name2 = 'Left', 'Right'
        isSame = int(isSame)
        image1 = np.transpose(image1,(1, 2, 0))
        image2 = np.transpose(image2,(1, 2, 0))
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        same = 1 if float(cos_fr) > threshold else 0
        title_str = getTFNPString(isSame, same)
        # Create visualization
        fig_size = (14, 3)
        # fig = plt.figure(tight_layout=True, figsize=fig_size)
        # fig = plt.figure(tight_layout=True)
        fig, axs = plt.subplots(1, 4, tight_layout=True, figsize=fig_size)
        fig.suptitle(title_str +
                     ' Cos=%.2f xCos=%.2f' % (float(cos_fr), cos_x))

        [axs[i].set_axis_off() for i in range(4)]
        # axs[0].text(0.5, 0.5, title_str + '\n Cos=%.2f\nxCos=%.2f'%(float(cos_fr), cos_x))
        axs[0].set_title('Face 1', y=-0.1)
        axs[1].set_title('Face 2', y=-0.1)
        axs[2].set_title(r'$cos_{patch}$', y=-0.1)
        axs[3].set_title(r'$weight_{attetion}$', y=-0.1)

        def drawGridLines(image_t, w_lines=5, h_lines=6,
                          colorRGB=(128, 128, 128)):
            '''
            colorRGB: default: gray(128, 128, 128), you can use red(255, 0, 0)
            '''
            colorRGB = (255, 0, 0)
            w_lines += 1
            h_lines += 1
            h, w, _ = image_t.shape
            w_unit = int(w // w_lines)
            # w_start = int(w_unit // 2)
            w_start = w_unit
            h_unit = int(h // h_lines)
            # h_start = int(h_unit // 2)
            h_start = h_unit
            # Draw vertical grid lines
            for step in range(w_lines):
                start_pt = (w_start + w_unit * step, 0)
                end_pt = (w_start + w_unit * step, h)
                cv2.line(image_t, start_pt, end_pt, colorRGB, 1, 1)
            # Draw horizontal grid lines
            for step in range(h_lines):
                start_pt = (0, h_start + h_unit * step)
                end_pt = (w, h_start + h_unit * step)
                cv2.line(image_t, start_pt, end_pt, colorRGB, 1, 1)
        drawGridLines(image1, 6, 6)
        drawGridLines(image2, 6, 6)
        axs[0].imshow(image1)
        axs[1].imshow(image2)
        # Show cos_patch
        im, cbar = heatmap_seaborn(cos_patch, [], [], ax=axs[2],
                           cmap="RdBu", threshold=threshold)
        # Show weights_attention
        im, cbar = heatmap(weight_attention, [], [], ax=axs[3],
                           cmap="YlGn")
        # texts = annotate_heatmap(im, valfmt="{x:.2f}")
        # axs[3].imshpw(cos_patch)
        # axs[4].imshow(weight_attention)
        # plt.show()
        img_name = exPath + '/' + title_str + \
            "_COS_%5.4f_xCos_%5.4f" % (float(cos_fr), cos_x) + '.png'
        plt.savefig(img_name, bbox_inches='tight')
        plt.gcf().clear()
        return

    # def find_lr(self,
    #             conf,
    #             init_value=1e-8,
    #             final_value=10.,
    #             beta=0.98,
    #             bloding_scale=3.,
    #             num=None):
    #     if not num:
    #         num = len(self.loader)
    #     mult = (final_value / init_value)**(1 / num)
    #     lr = init_value
    #     for params in self.optimizer_fr.param_groups:
    #         params['lr'] = lr
    #     for params in self.optimizer_atten.param_groups:
    #         params['lr'] = lr
    #     self.model.train()
    #     avg_loss = 0.
    #     best_loss = 0.
    #     batch_num = 0
    #     losses = []
    #     log_lrs = []
    #     # TODO
    #     for i, (img1s, img2s, label1s, label2s) in tqdm(enumerate(self.loader), total=num):

    #         imgs = imgs.to(conf.device)
    #         labels = labels.to(conf.device)
    #         batch_num += 1

    #         self.optimizer_fr.zero_grad()
    #         self.optimizer_atten.zero_grad()

    #         embeddings = self.model(imgs)
    #         thetas = self.head(embeddings, labels)
    #         loss = conf.loss_fr(thetas, labels)

    #         # Compute the smoothed loss
    #         avg_loss = beta * avg_loss + (1 - beta) * loss.item()
    #         self.writer.add_scalar('avg_loss', avg_loss, batch_num)
    #         smoothed_loss = avg_loss / (1 - beta**batch_num)
    #         self.writer.add_scalar('smoothed_loss', smoothed_loss, batch_num)
    #         # Stop if the loss is exploding
    #         if batch_num > 1 and smoothed_loss > bloding_scale * best_loss:
    #             print('exited with best_loss at {}'.format(best_loss))
    #             plt.plot(log_lrs[10:-5], losses[10:-5])
    #             return log_lrs, losses
    #         # Record the best loss
    #         if smoothed_loss < best_loss or batch_num == 1:
    #             best_loss = smoothed_loss
    #         # Store the values
    #         losses.append(smoothed_loss)
    #         log_lrs.append(math.log10(lr))
    #         self.writer.add_scalar('log_lr', math.log10(lr), batch_num)
    #         # Do the SGD step
    #         # Update the lr for the next step

    #         loss.backward()
    #         self.optimizer_fr.step()
    #         self.optimizer_atten.step()

    #         lr *= mult
    #         for params in self.optimizer_fr.param_groups:
    #             params['lr'] = lr
    #         for params in self.optimizer_atten.param_groups:
    #             params['lr'] = lr
    #         if batch_num > num:
    #             plt.plot(log_lrs[10:-5], losses[10:-5])
    #             return log_lrs, losses

    def train(self, conf, epochs, resume=True):
        self.load_state(conf, conf.fixed_str,
                        from_save_folder=True,
                        model_only=True,
                        strict=False,
                        model_atten=False)
        self.model.train()
        self.model_attention.train()
        # print('>>> Initialize, testing xCos on lfw')
        # accuracy, best_threshold, roc_curve_tensor = self.evaluate_attention(conf, self.lfw, self.lfw_issame)
        # self.board_val('lfw_xCos', accuracy, best_threshold, roc_curve_tensor)
        # self.model.train()
        # self.model_attention.train()
        # print('>>>> done testing, xCos accuracy on LFW:', accuracy)
        running_loss = 0.
        for e in range(epochs):
            print('epoch {} started'.format(e))
            self.scheduler_atten.step()
            if e == self.milestones[0]:
                self.schedule_lr()
            if e == self.milestones[1]:
                self.schedule_lr()
            if e == self.milestones[2]:
                self.schedule_lr()
            # TODO
            # for imgs, labels in tqdm(iter(self.loader)):
            for img1s, img2s, label1s, label2s in tqdm(self.loader, total=len(self.loader)):
                '''
                img1s: [bs, 3, 112, 112]
                label1s: [bs]
                '''
                img1s = img1s.to(conf.device)
                img2s = img2s.to(conf.device)
                label1s = label1s.to(conf.device)
                label2s = label2s.to(conf.device)

                imgs = torch.cat((img1s, img2s), 0)
                labels = torch.cat((label1s, label2s), 0)

                self.optimizer_fr.zero_grad()
                self.optimizer_atten.zero_grad()

                # Part1: FR
                embeddings, grid_feats = self.model(imgs)
                if conf.modelType == 'SphereFace':
                    thetas = self.head(embeddings)
                else:
                    thetas = self.head(embeddings, labels)
                loss1 = self.loss_fr(thetas, labels)

                # Part2: xCos
                cos_gts = self.getCos(imgs)
                half_idx = grid_feats.size(0) // 2
                grid_feat1s = grid_feats[:half_idx]
                grid_feat2s = grid_feats[half_idx:]
                attention = self.model_attention(grid_feat1s, grid_feat2s)

                if conf.detachAttentionGradient:
                    grid_feat1s = grid_feat1s.detach()
                    grid_feat2s = grid_feat2s.detach()

                loss2 = self.xCos_loss_with_attention(grid_feat1s,
                                                      grid_feat2s,
                                                      attention,
                                                      cos_gts)
                # TODO alpha weight
                alpha = 0.5
                loss = alpha * loss1 + (1 - alpha) * loss2
                loss.backward()
                running_loss += loss.item()
                self.optimizer_fr.step()
                self.optimizer_atten.step()

                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar(self.conf.exp_title + '/train_loss', loss_board, self.step)
                    self.writer.add_scalar(self.conf.exp_title + '/loss_fr', loss1.item(), self.step)
                    self.writer.add_scalar(self.conf.exp_title + '/loss_atten', loss2.item(), self.step)
                    running_loss = 0.

                if self.step % self.evaluate_every == 0 and self.step != 0:
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.agedb_30, self.agedb_30_issame)
                    self.board_val('agedb_30', accuracy, best_threshold, roc_curve_tensor)
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.lfw, self.lfw_issame)
                    self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor)
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.cfp_fp, self.cfp_fp_issame)
                    self.board_val('cfp_fp', accuracy, best_threshold, roc_curve_tensor)
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate_attention(conf, self.lfw, self.lfw_issame)
                    self.board_val('lfw_xCos', accuracy, best_threshold, roc_curve_tensor)
                    self.model.train()
                    self.model_attention.train()
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy, extra=conf.modelType)

                self.step += 1

        self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def schedule_lr(self):
        for params in self.optimizer_fr.param_groups:
            params['lr'] /= 10
        print(self.optimizer_fr)

    def infer(self, conf, faces, target_embs, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if tta:
                mirror = trans.functional.hflip(img)
                emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:
                embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        source_embs = torch.cat(embs)

        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
        return min_idx, minimum
