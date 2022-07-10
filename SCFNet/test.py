import sys
sys.dont_write_bytecode = True
import cv2
import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader
from lib import dataset
from multiprocessing import Process
from saliency_toolbox import calculate_measures
import warnings
warnings.filterwarnings('ignore')


class Test(object):
    def __init__(self, Dataset, datapath, Network, model_path):
        self.datapath = datapath.split("/")[-1]
        print(datapath)
        self.cfg = Dataset.Config(datapath=datapath, mode='test')
        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(
            self.data,
            batch_size=1,
            shuffle=False,
            num_workers=4
        )
        # network
        self.net = Network
        self.net.eval()
        self.net.load_dict(paddle.load(model_path))

    def save(self, save_path):
        save_paths = os.path.join(save_path, self.cfg.datapath.split('/')[-1])
        if not os.path.exists(save_paths):
            os.makedirs(save_paths)
        with paddle.no_grad():
            for num, (image, mask, (H, W), maskpath) in enumerate(self.loader):
                out = self.net(image)
                pred = F.sigmoid(out[0])
                pred = F.interpolate(pred, size=(H[0], W[0]), mode='bilinear', align_corners=True)
                pred = pred[0].transpose((1, 2, 0)) * 255
                cv2.imwrite(save_paths + '/' + maskpath[0] + '.png', pred.cpu().numpy())


def test_socre(path, save_root=None):
    if save_root:
        if not os.path.exists(save_root):
            os.mkdir(save_root)
    sms_dir = [
        path + '/DUTS-TE/',
        path + '/ECSSD/',
        path + 'HKU-IS/',
        path + '/DUT-OMRON/',
        path + '/PASCAL-S/'
    ]
    gts_dir = [
        './data/DUTS-TE/mask'
        './data/ECSSD/mask',
        './data/HKU-IS',
        './data/DUT-OMRON/mask',
        './data/PASCAL-S/mask',
    ]
    measures = ['Max-F', 'MAE', 'E-measure', 'Wgt-F']
    for i in range(len(gts_dir)):
        if save_root:
            save = save_root + '/' + sms_dir[i].split('/')[-2]
            print(save)
        else:
            save=None
        res = calculate_measures(gts_dir[i], sms_dir[i], measures, save=save)
        print(path, gts_dir[i].split('/')[-3], 'MAE:', res['MAE'], 'Fm:', res['Mean-F'], 'E-measure:',
              res['E-measure'], 'Wgt-F', res['Wgt-F'])


def mutil_test_score(path, save_root=None):
    '''
    param1: the parent directory of the predicted saliency maps for the 5 datasets. 
    param2: the address where the metrics results are saved.
    '''
    
    if save_root:
        if not os.path.exists(save_root):
            os.makedirs(save_root)
    sms_dir = [
       path + '/DUTS-TE/',
        path + '/ECSSD/',
       path + '/HKU-IS/',
       path + '/DUT-OMRON/',
       path + '/PASCALS/'
    ]
    gts_dir = [
       './data/DUTS-TE/mask/',
       './data/ECSSD/mask/',
       './data/HKU-IS/mask/',
       './data/DUT-OMRON/mask/',
       './data/PASCAL-S/mask/'
    ]
    measures = ['Max-F', 'MAE', 'E-measure', 'Wgt-F']
    if save_root is not None:
        saves = [save_root + '/' + sms_dir[i].split('/')[-2] for i in range(len(sms_dir))]
    else:
        saves = [None] * len(sms_dir)

    processes = [Process(target=sing_score, args=(path, gts_dir[i], sms_dir[i], measures, saves[i]), ) for i in range(len(sms_dir))]
    [p.start() for p in processes]


def sing_score(path, gt, pre, measures, save=None):
    print(gt)
    print(pre)
    res = calculate_measures(gt, pre, measures, save=save)
    print(path, gt.split('/')[-3], 'MAE:', res['MAE'], 'Fm:', res['Mean-F'], 'E-measure:',
          res['E-measure'], 'Wgt-F', res['Wgt-F'])


def save_results(model, model_path, save=None):

    DATASETS = [
      './data/PASCAL-S',
      './data/ECSSD',
      './data/DUT-OMRON',
      './data/HKU-IS',
      './data/DUTS-TE',
    ]
    print(model_path)
    for e in DATASETS:
        t = Test(dataset, e, model, model_path)
        t.save(save_path=save)


if __name__=='__main__':
    import os
    from net import SCFNet
    model_path = "./weight/KD-SCFNet/KD-SCFNet.pdparams"
    save_results(SCFNet(1, 'M3_0.5'), model_path, save='./maps/KD-SCFNet')
    mutil_test_score('./maps/KD-SCFNet', './scores/KD-SCFNet')
