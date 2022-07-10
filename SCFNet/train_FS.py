import datetime
import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader
from lib import dataset
import numpy as np
import cv2
import argparse
import sys
import os
import random
from visualdl import LogWriter


# config
def config():
    parser = argparse.ArgumentParser(description='train params')
    parser.add_argument('--Min_LR', default=0.00064, help='min lr')
    parser.add_argument('--Max_LR', default=0.064)
    parser.add_argument('--model_type', default='M3_0.5')
    parser.add_argument('--top_epoch', default=5)
    parser.add_argument('--epoch', default=69)
    parser.add_argument('--snapshot', default=None, help='where your pretrained model')
    parser.add_argument('--train_bs', default=128)
    parser.add_argument('--decay', default=5e-4)
    parser.add_argument('--momen', default=0.9)
    parser.add_argument('--train_mode', default='fs')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--show_step', default=30)
    parser.add_argument('--datapath', default=r'./data/DUTS-TR')
    parser.add_argument('--savepath', default='./weight/FS-SCFNet')
    parser.add_argument('--save_iter', default=1, help=r'every iter to save model')
    parser.add_argument('--log_dir', default='./log/FS-SCFNet')
    cag = parser.parse_args()
    return cag


cag = config()


# lr scheduler
def lr_decay(steps, scheduler):
    mum_step = cag.top_epoch * (10553 / cag.train_bs + 1)
    min_lr = cag.Min_LR
    max_lr = cag.Max_LR
    total_steps = cag.epoch * (10553 / cag.train_bs + 1)
    if steps < mum_step:
        lr = min_lr + abs(max_lr - min_lr) / (mum_step) * steps
    else:
        lr = scheduler.get_lr()
        scheduler.step()
    return lr


# dice loss
def dice_loss(pred, mask):
    pred          = F.sigmoid(pred)
    intersection  = (pred * mask).sum(axis=(2, 3))
    unior         = (pred + mask).sum(axis=(2, 3))
    dice          = (2 * intersection + 1) / (unior + 1)
    dice          = paddle.mean(1 - dice)
    return dice


# boundary dice loss and PBSM
def boundary_dice_loss(pred, mask, epoch):
    size = max(1, (13 - ((epoch + 1) // 10) * 2))  # PBSM
    pred = F.sigmoid(pred)
    n    = pred.shape[0]
    mask_boundary = paddle.nn.functional.max_pool2d(1 - mask, kernel_size=3, stride=1, padding=1)
    mask_boundary -= 1 - mask

    pred_boundary = paddle.nn.functional.max_pool2d(1 - pred, kernel_size=3, stride=1, padding=1)
    pred_boundary -= 1 - pred
    mask_boundary = paddle.nn.functional.max_pool2d(mask_boundary, kernel_size=size, stride=1, padding=(size-1)//2)
    pred_boundary = paddle.nn.functional.max_pool2d(pred_boundary, kernel_size=size, stride=1, padding=(size-1)//2)
    mask_boundary = paddle.reshape(mask_boundary, shape=(n, -1))
    pred_boundary = paddle.reshape(pred_boundary, shape=(n, -1))

    intersection  = (pred_boundary * mask_boundary).sum(axis=(1))
    unior         = (pred_boundary + mask_boundary).sum(axis=(1))
    dice          = (2 * intersection + 1) / (unior + 1)
    dice          = paddle.mean(1 - dice)
    return dice


# train
def train(Dataset, Network):
    # dataset
    if not os.path.exists(cag.savepath):
        os.makedirs(cag.savepath)
    
    data = Dataset.Data(cag)
    loader = DataLoader(
        data,
        batch_size=cag.train_bs,
        shuffle=True,
        num_workers=4,
        collate_fn=data.collate
    )

    # network 
    '''
    Change the second parameter to change different networks, for example: "R50" means SCFNet with ResNet-50 as backbone, "M3_0.5" 
    means SCFNet with MobileNetV3-0.5 as the backbone
    '''
    net = Network(cag)
    net.train()

    # params
    total_params = sum(p.numel() for p in net.parameters())
    print('total params : ', total_params)

    # optimizer
    optimizer = paddle.optimizer.Momentum(parameters=net.parameters(), learning_rate=cag.Max_LR, momentum=cag.momen,
                                          weight_decay=cag.decay)
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=cag.Max_LR,
                                                         T_max=len(loader) * (cag.epoch-cag.top_epoch), eta_min = cag.Min_LR)
    global_step = 0
    epoch_step = len(loader)

    # training
    with LogWriter(logdir=cag.log_dir) as writter:
        for epoch in range(0, cag.epoch):
            start = datetime.datetime.now()
            for batch_idx, (image, mask) in enumerate(loader, start=1):
                lr = lr_decay(global_step, scheduler)
                optimizer.clear_grad()
                optimizer.set_lr(lr)

                writter.add_scalar(tag='train/lr', step=global_step, value=lr)
                global_step += 1
                out2, out3, out4, out5 = net(image)
                loss1 = dice_loss(out2, mask) + 1 * boundary_dice_loss(out2, mask, epoch)
                loss2 = dice_loss(out3, mask) + 1 * boundary_dice_loss(out3, mask, epoch)
                loss3 = dice_loss(out4, mask) + 1 * boundary_dice_loss(out4, mask, epoch)
                loss4 = dice_loss(out5, mask) + 1 * boundary_dice_loss(out5, mask, epoch)
                loss = loss1 + loss2 / 2 + loss3 / 4 + loss4 / 8
                loss.backward()
                optimizer.step()

                # log
                writter.add_scalar(tag='train/loss', step=global_step, value=loss.numpy()[0])
                writter.add_scalar(tag='train/loss1', step=global_step, value=loss1.numpy()[0])
                writter.add_scalar(tag='train/loss2', step=global_step, value=loss2.numpy()[0])
                writter.add_scalar(tag='train/loss3', step=global_step, value=loss3.numpy()[0])
                writter.add_scalar(tag='train/loss4', step=global_step, value=loss4.numpy()[0])
                
                # output log
                if batch_idx % cag.show_step == 0:
                    msg = '%s | step:%d/%d/%d (%.2f%%) | lr=%.6f |  loss=%.6f | loss1=%.6f | loss2=%.6f | loss3=%.6f |  loss3=%.6f | %s ' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), batch_idx, epoch + 1, cag.epoch, 
                    batch_idx / (10553 / cag.train_bs) * 100, optimizer.get_lr(), loss.item(), loss1.item(),
                        loss2.item(), loss3.item(), loss4.item(), image.shape)
                    print(msg)

            # save weight
            if epoch > cag.epoch / 3 * 2:
                paddle.save(net.state_dict(), cag.savepath + '/model-' + str(epoch + 1) + '.pdparams')

            # ETA
            end = datetime.datetime.now()
            spend = int((end - start).seconds)
            eta = datetime.timedelta(seconds = spend * (cag.epoch - epoch))
            eta = datetime.datetime.now() + eta  
            mins = spend // 60
            secon = spend % 60
            print(f'this epoch spend {mins} m {secon} s, eta: {eta.strftime("%Y-%m-%d %H:%M:%S")}. \n')

            
if __name__ == '__main__':
    from net import SCFNet
    train(dataset, SCFNet)
