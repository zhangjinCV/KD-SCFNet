import os
import os.path as osp
import cv2
import paddle
import numpy as np
from paddle.io import Dataset
from lib.transform import Normalize, RandomBlur, RandomCrop, RandomHorizontalFlip, RandomVorizontalFlip, Resize, ToTensor, RandomBrightness
import random
import matplotlib
matplotlib.use('TkAgg')


class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


class Data(Dataset):
    def __init__(self, cfg):
        super(Data, self).__init__()

        self.cfg = cfg
        self.blur        = RandomBlur(0.1)
        self.brightness  = RandomBrightness()
        self.normalize   = Normalize()
        self.randomcrop  = RandomCrop()
        self.randomvflip = RandomVorizontalFlip()
        self.randomhflip = RandomHorizontalFlip()
        self.resize      = Resize(352, 352)
        self.totensor    = ToTensor()
        with open(cfg.datapath+'/'+cfg.mode+'.txt', 'r') as lines:
            self.samples = []
            for line in lines:
                self.samples.append(line.strip())

    def __getitem__(self, idx):
        name = self.samples[idx]
        image = cv2.imread(self.cfg.datapath+'/image/' + name + '.jpg')[:,:,::-1].astype(np.float32)
        if self.cfg.train_mode == 'kd':
            mask = None
        else:
            mask = cv2.imread(self.cfg.datapath + '/mask/' + name + '.png')[:, :, ::-1].astype(np.float32)
        H, W, C = image.shape
        if self.cfg.mode == 'train':
            image, mask = self.blur(image, mask)
            image, mask = self.brightness(image, mask)
            image, mask = self.normalize(image, mask)
            image, mask = self.randomcrop(image, mask)
            image, mask = self.randomhflip(image, mask)
            image, mask = self.randomvflip(image, mask)
            return image, mask
        else:
            image, mask = self.normalize(image, mask)
            image, mask = self.resize(image, mask)
            image, mask = self.totensor(image, mask)
            return image, mask, (H, W), name

    def __len__(self):
        return len(self.samples)

    def collate(self, batch):
        size = [288, 256, 320, 352, 224][np.random.randint(0, 5)]
        image, mask = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            if self.cfg.train_mode != 'kd':
                mask[i] = cv2.resize(mask[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        image = (np.stack(image, axis=0)).transpose((0, 3, 1, 2)) # bs c h w
        if self.cfg.train_mode != 'kd':
            mask = (np.stack(mask, axis=0)).transpose((0, 3, 1, 2)) # bs c h w
            mask = np.mean(mask, axis=1, keepdims=True) # bs c h w
            mask = mask.astype(np.float32)
        return image.astype(np.float32), mask


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from paddle.io import DataLoader

    plt.ion()
    cfg = Config(mode='train', datapath=r'F:\Dataset\DUTS\DUTS-TR')
    data = Data(cfg)
    loader = DataLoader(data, batch_size=8, shuffle=True, collate_fn=data.collate)
    for step, (image, mask) in enumerate(loader):
        print(image.shape)
        image = image[0].numpy()
        mask = mask[0]
        mask = mask * 255
        mask = mask.numpy().squeeze()
        image = image.transpose((1, 2, 0))
        image = image * cfg.std + cfg.mean
        plt.subplot(121)
        plt.imshow(np.uint8(image))
        plt.subplot(122)
        plt.imshow(mask, cmap='binary')
        plt.show()
        input()
