import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision.datasets import ImageFolder
cos = nn.CosineSimilarity(dim=0, eps=1e-6)


class myImageFolder(ImageFolder):
    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    def __init__(self, root, transform=None, target_transform=None):
        super(myImageFolder, self).__init__(root, transform, target_transform)


class SiameseImageFolder(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, imgs_folder_dir, transform):
        print('>>> In SIFolder, imgfolderdir=', imgs_folder_dir)
        self.root = imgs_folder_dir
        self.wFace_dataset = ImageFolder(imgs_folder_dir, transform)
        self.class_num = len(self.wFace_dataset.classes)
        print('>>> self.class_num = ', self.class_num)
        # self.memoryAll = False

        self.train_labels = np.array(self.wFace_dataset.targets, dtype=int)
        print('>>> self.train_labels:', self.train_labels[1000:1010])

        self.train_data = self.wFace_dataset

        # XXX
        # if self.memoryAll:
        #    self.train_data = self.wFace_dataset.train_data

        self.labels_set = set(self.train_labels)
        self.label_to_indices = {label:
                                 np.where(self.train_labels
                                          == label)[0]
                                 for label in self.labels_set}
        print('>>> Init SiameseImageFolder done!')

    def __getitem__(self, index):
        '''
        img1 = (feat_fc, feat_grid)
        '''
        target = np.random.randint(0, 2)
        img1, label1 = self.train_data[index]  # , self.train_labels[index].item()
        if target == 1:
            siamese_index = index
            while siamese_index == index:
                siamese_index = np.random.choice(self.label_to_indices[label1])
        else:
            siamese_label = np.random.choice(
                    list(self.labels_set - set([label1])))
            siamese_index = np.random.choice(
                    self.label_to_indices[siamese_label])
        img2, label2 = self.train_data[siamese_index]

        # XXX stack
        # stack (img1, img2), (label1, label2), cos_gt
        return img1, img2, label1, label2

    def __len__(self):
        return len(self.wFace_dataset)


class SiameseWholeFace(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """
    # @property
    # def train_data(self):
    #     warnings.warn("train_data has been renamed data")
    #     return self.wFace_dataset

    # @property
    # def test_data(self):
    #     warnings.warn("test_data has been renamed data")
    #     return self.wFace_dataset

    def __init__(self, wFace_dataset):
        self.wFace_dataset = wFace_dataset
        self.train = self.wFace_dataset.train
        self.memoryAll = self.wFace_dataset.memoryAll

        if self.train:
            self.train_labels = self.wFace_dataset.train_labels
            self.train_data = self.wFace_dataset
            if self.memoryAll:
                self.train_data = self.wFace_dataset.train_data

            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label:
                                     np.where(self.train_labels.numpy()
                                              == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            # TODO: @property like MNIST
            self.test_labels = self.wFace_dataset.test_labels
            self.test_data = self.wFace_dataset
            if self.memoryAll:
                self.test_data = self.wFace_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label:
                                     np.where(self.test_labels.numpy()
                                              == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs
        print('>>> Init SiameseWholeFace done!')
    def __getitem__(self, index):
        '''
        img1 = (feat_fc, feat_grid)
        '''
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]
        # [Depreciated] feat1 1 is of size [21504]
        # feat1, feat2 = img1.view(-1), img2.view(-1)
        # cosine = cos(feat1, feat2).numpy()
        # target = cosine
        feat_grid_1 , feat_fc_1 = img1
        feat_grid_2 , feat_fc_2 = img2
        return (feat_grid_1, feat_fc_1, feat_grid_2, feat_fc_2), target

    def __len__(self):
        return len(self.wFace_dataset)


class SiameseENM(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, ENM_dataset):
        self.ENM_dataset = ENM_dataset
        self.train = self.ENM_dataset.train
        # self.train = False

        if self.train:
            self.train_labels = self.ENM_dataset.train_labels
            self.train_data = self.ENM_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label:
                                     np.where(self.train_labels.numpy()
                                              == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            # TODO: @property like MNIST
            self.test_labels = self.ENM_dataset.test_labels
            self.test_data = self.ENM_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label:
                                     np.where(self.test_labels.numpy()
                                              == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        return (img1, img2), target

    def __len__(self):
        return len(self.ENM_dataset)


class TripletENM(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, ENM_dataset):
        self.ENM_dataset = ENM_dataset
        self.train = self.ENM_dataset.train

        if self.train:
            self.train_labels = self.ENM_dataset.train_labels
            self.train_data = self.ENM_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.ENM_dataset.test_labels
            self.test_data = self.ENM_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        return (img1, img2, img3), []

    def __len__(self):
        return len(self.ENM_dataset)


class SiameseMNIST(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.mnist_dataset)


class TripletMNIST(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
import os
import pickle
import torch
import warnings
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from joblib import Parallel, delayed


class ENMDataset:
    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root_dir, featType='nose', train=True, valSize=0.2):
        print('>>> Init ENM dataset for', featType,
              root_dir, 'isTrain:', train)
        self.featType = featType
        self.id_paths = glob(os.path.join(root_dir, '*/'))
        dataPaths = [glob(d + '*.pkl') for d in self.id_paths]
        dataPaths = [item for sublist in dataPaths for item in sublist]
        train_ps, val_ps = train_test_split(dataPaths,
                                            test_size=valSize,
                                            random_state=42)
        self.train_id_paths, self.val_id_paths = train_ps, val_ps
        # print(len(self.train_id_paths), len(self.val_id_paths))
        # print(self.train_id_paths[:4])
        # print(self.val_id_paths[:4])
        self.train = train
        if self.train:
            dataPaths = self.train_id_paths
        else:
            dataPaths = self.val_id_paths
        # print(len(dataPaths), dataPaths[:4])
        trainFlag = 'train' if self.train else 'val'
        self.pkl_fname = os.path.join(config.FEAT_DIR, self.featType + '_' + trainFlag + '.pkl')
        if os.path.isfile(self.pkl_fname):
            print('>>> Loading:', self.pkl_fname)
            with open(self.pkl_fname, 'rb') as h:
                self.data, self.targets = pickle.load(h)
        else:
            self.data, self.targets = self.loadData(dataPaths)

    def loadData(self, paths):
        # XXX train set: 12 min; val set: 3 min
        data = []
        targets = []
        def loadPkl(p):
            with open(p, 'rb') as h:
                feat = pickle.load(h)
                data.append(feat[self.featType])
                targets.append(feat['label'])
        Parallel(n_jobs=config.WORKERS, require='sharedmem', prefer="threads")(delayed(loadPkl)(p) for p in tqdm(paths))

        # for p in tqdm(paths):
        #     with open(p, 'rb') as h:
        #         feat = pickle.load(h)
        #         data.append(feat[self.featType])
        #         targets.append(feat['label'])

        # print(data[0])
        # print(targets[0])
        data = torch.stack(data, 0)
        targets = torch.stack(targets, 0)
        # print(data.size())
        # print(targets.size())

        # Save pkl
        with open(self.pkl_fname, 'wb') as h:
            pickle.dump((data, targets), h, protocol=pickle.HIGHEST_PROTOCOL)

        return data, targets

    def __getitem__(self, key):  # XXX It seems useless?
        return self.data[key], self.targets[key]

    def __len__(self):
        return len(self.data)


class FaceFeatDataset:
    # XXX: self.target contain all labels;
    #      while _getitem_ can only get image feat
    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data_mix

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data_mix

    def __init__(self, root_dir, train=True, valSize=0.2, memoryAll=False):
        print('>>> Init FaceFeat dataset for', root_dir, 'isTrain:', train, 'memoryAll:', memoryAll)
        self.memoryAll = memoryAll
        self.id_paths = glob(os.path.join(root_dir, '*/'))
        dataPaths = [glob(d + '*.pkl') for d in self.id_paths]
        dataPaths = [item for sublist in dataPaths for item in sublist]
        train_ps, val_ps = train_test_split(dataPaths,
                                            test_size=valSize,
                                            random_state=42)
        self.train_id_paths, self.val_id_paths = train_ps, val_ps
        # print(len(self.train_id_paths), len(self.val_id_paths))
        # print(self.train_id_paths[:4])
        # print(self.val_id_paths[:4])
        self.train = train
        if self.train:
            self.dataPaths = self.train_id_paths
        else:
            self.dataPaths = self.val_id_paths
        trainFlag = 'train' if self.train else 'val'

        # Load target labels
        if not self.memoryAll:
            self.pkl_fname = os.path.join(config.FEAT_DIR, 'target_' + trainFlag + '.pkl')
            if os.path.isfile(self.pkl_fname):
                print('>>> Loading:', self.pkl_fname)
                with open(self.pkl_fname, 'rb') as h:
                    self.targets = pickle.load(h)
            else:
                self.targets = self.loadTargets(self.dataPaths)
        else:
            self.pkl_fname = os.path.join(config.FEAT_DIR, 'targetAndData_' + trainFlag + '.pkl')
            if os.path.isfile(self.pkl_fname):
                print('>>> Loading:', self.pkl_fname)
                with open(self.pkl_fname, 'rb') as h:
                    self.data, self.data_fc, self.targets = pickle.load(h)
                    self.data_mix = DataMix(self.data, self.data_fc)
            else:
                self.data, self.data_fc, self.targets = self.loadData(self.dataPaths)
                self.data_mix = DataMix(self.data, self.data_fc)

        print('>>> Init done!')

    def loadData(self, paths):
        # XXX train set: 12 min; val set: 3 min
        data = []
        data_fc = []
        targets = []
        def loadPkl(p):
            with open(p, 'rb') as h:
                feat = pickle.load(h)
                data.append(feat['wholeFaceFeat_42grid'])
                data_fc.append(feat['wholeFaceFeat'])
                targets.append(feat['label'])
                del feat
        print('>>> In faceFeatDataset, Loading ALL datas')
        Parallel(n_jobs=6, require='sharedmem', prefer="threads")(delayed(loadPkl)(p) for p in tqdm(paths))

        # for p in tqdm(paths):
        #     with open(p, 'rb') as h:
        #         feat = pickle.load(h)
        #         data.append(feat[self.featType])
        #         targets.append(feat['label'])

        # print(data[0])
        # print(targets[0])
        data = torch.stack(data, 0)
        data_fc = torch.stack(data_fc, 0)
        targets = torch.stack(targets, 0)
        # print(data.size())
        # print(targets.size())

        # Save pkl
        with open(self.pkl_fname, 'wb') as h:
            pickle.dump((data, data_fc, targets), h, protocol=pickle.HIGHEST_PROTOCOL)

        return data, data_fc, targets

    def loadPkl(self, p):
        with open(p, 'rb') as h:
            feat = pickle.load(h)
            return feat['wholeFaceFeat_42grid'], feat['wholeFaceFeat'], feat['label']

    def loadFeat(self, p):
        with open(p, 'rb') as h:
            feat = pickle.load(h)
            return feat['wholeFaceFeat_42grid'], feat['wholeFaceFeat']

    def loadTargets(self, paths):
        # XXX train set: 12 min; val set: 3 min
        targets = []

        def loadpkl(p):
            with open(p, 'rb') as h:
                feat = pickle.load(h)
                targets.append(feat['label'])
        Parallel(n_jobs=config.WORKERS, require='sharedmem', prefer="threads")(delayed(loadpkl)(p) for p in tqdm(paths))

        # print(targets[0])
        targets = torch.stack(targets, 0)
        # print(targets.size())

        # Save pkl
        with open(self.pkl_fname, 'wb') as h:
            pickle.dump(targets, h, protocol=pickle.HIGHEST_PROTOCOL)

        return targets

    def __getitem__(self, key):
        # XXX
        raise NotImplementedError
        if not self.memoryAll:
            return self.loadFeat(self.dataPaths[key])
        else:
            return self.data[key], self.data_fc[key], self.targets[key]

    def __len__(self):
        return len(self.dataPaths)


class DataMix:
    def __init__(self, data_grid, data_fc):
        self.data_grid = data_grid
        self.data_fc = data_fc
        self._data_len = len(self.data_grid)

    def __getitem__(self, key):
        return (self.data_grid[key], self.data_fc[key])

    def __len__(self):
        return self._data_len


if __name__ == "__main__":
    with torch.no_grad():
        # dataset = ENMDataset(config.FEAT_DIR, train=False, valSize=0.05)
        dataset = FaceFeatDataset(config.FEAT_DIR, train=False, valSize=0.2)
        print(dataset.train)
        print(dataset)
        # dataset1 = FaceFeatDataset(config.FEAT_DIR, train=True, valSize=0.2)
        siamese_val_dataset = SiameseWholeFace(dataset)
        (img1, img2), label = siamese_val_dataset[-1]
        print(img1.size(), img2.size(), label)
        for i in range(len(siamese_val_dataset)):
            if i == 5: break
            (img1, img2), label = siamese_val_dataset[i]
            print(img1.size(), img2.size(), label)
