# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
from tqdm import tqdm

import torch
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate

import numpy as np
import cv2
from numpy.linalg import norm, norm









def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc  = AverageMeter()
    avg_sem_loss = AverageMeter()
    avg_bce_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, batch in enumerate(trainloader, 0):
        images, labels, bd_gts, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()
        bd_gts = bd_gts.float().cuda()
        

        losses, _, acc, loss_list = model(images, labels, bd_gts)
        loss = losses.mean()
        acc  = acc.mean()

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())
        ave_acc.update(acc.item())
        avg_sem_loss.update(loss_list[0].mean().item())
        avg_bce_loss.update(loss_list[1].mean().item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, Acc:{:.6f}, Semantic loss: {:.6f}, BCE loss: {:.6f}, SB loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average(),
                      ave_acc.average(), avg_sem_loss.average(), avg_bce_loss.average(),ave_loss.average()-avg_sem_loss.average()-avg_bce_loss.average())
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1






def validate(config, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, label, bd_gts, _, _ = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            bd_gts = bd_gts.float().cuda()

            losses, pred, _, _ = model(image, label, bd_gts)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            if idx % 10 == 0:
                print(idx)

            loss = losses.mean()
            ave_loss.update(loss.item())

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        
        logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array












from numpy.linalg import norm

from scipy.spatial import distance

import math

def find_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def similarity(x, y):
    v = np.dot(x, y) / (norm(x) * norm(y))
    return v


def fire_angle(b):
    return np.arctan((b[0] + b[3]) * 1.0 / (b[1] + b[2] + 1e-8)) * 180 / np.pi


def maximum_filter(n, img):
    # Creates the shape of the kernel
    size = (n, n)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)

    # Applies the maximum filter with kernel NxN
    imgResult = cv2.dilate(img, kernel)
    # kernel = cv2.getStructuringElement(shape, size)

    imgResult = cv2.erode(imgResult, kernel)
    return imgResult


def distances(centers):
    distance = []

    for i in range(len(centers)):
        for j in range(i+1,len(centers)):
            if j!=i:
                distance.append(find_distance(centers[i], centers[j]))

    return distance

import cv2

def cosine_similarity(mask,pred):
    mask = np.sort(mask,axis=0)
    pred = np.sort(pred,axis=0)
    if mask.shape[1] > 1:
        mm = np.concatenate(mask).ravel()
    else:
        mask.ravel()
    if pred.shape[1] > 1:
        pp = np.concatenate(pred).ravel()
    else:
        pp=pred.ravel()
    mlength = max(len(mm),len(pp))
    mm = np.pad(mm,((0, mlength - mm.shape[0])), mode='constant')
    pp = np.pad(pp,((0, mlength - pp.shape[0])), mode='constant')
    v = np.dot(mm,pp)/(norm(mm)*norm(pp))
    return v

def extract_metrics(mat):
    mat = np.asarray(mat,dtype='uint8')

    contours, hierarchy = cv2.findContours(mat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 2, True)
        boundRect[i] = np.asarray(cv2.boundingRect(contours_poly[i]))
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

    blur = maximum_filter(17, mat)

    number_of_fires, _ = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    deviation = np.std(distances(centers))
    if len(centers) <= 1:
        deviation = 0.0

    areas = np.zeros(len(number_of_fires))

    for j, c in enumerate(number_of_fires):
        areas[j] = cv2.contourArea(c)
    if len(number_of_fires)==0:
        areas = np.asarray([0.0])
    return boundRect,len(number_of_fires),deviation,areas

def normalize(a,b):
    return np.abs(a-b)

def normalize_percent(a,b):
    if(max(norm(a),norm(b)))!=0.0:
        return min(norm(a),norm(b))*1.0/max(norm(a),norm(b))
    return 0.0

def testval(config, test_dataset, testloader, model,
            sv_dir, sv_pred=True):
    model.eval()
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))

    areas = np.zeros((len(tqdm(testloader)),))
    no_fires = np.zeros((areas.shape))
    deviations = np.zeros((areas.shape))
    areas_norm = np.zeros((areas.shape))
    no_fires_norm = np.zeros((areas.shape))
    deviations_norm = np.zeros((areas.shape))
    cosine_sim = np.zeros((areas.shape))
    no_fires_arr = np.zeros((areas.shape[0],2))

    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, _, name = batch
            size = label.size()
            pred = test_dataset.single_scale_inference(config, model, image.cuda())

            im = image.cpu()
            #orig = np.zeros((im.shape[2], im.shape[3]))
            #orig = np.asarray(im[0, 1, :, :], dtype ='uint8')

            cp = pred.cpu()
            #mat = np.zeros((cp.shape[2], cp.shape[3]))
            mat = np.asarray(cp[0, 1, :, :],dtype = 'uint8')


            lcp = label.cpu()
            #ll = np.zeros((lcp.shape[1],lcp.shape[2]))
            ll = np.asarray(lcp[0,:,:],dtype='uint8')

            m,n,o,k = extract_metrics(mat)
            p,q,r,l = extract_metrics(ll)

            area_l_mean = l.mean()

            area_k_mean = k.mean()

            areas[index] = normalize(area_l_mean, area_k_mean)
            deviations[index] = normalize(r, o)
            no_fires[index] = np.abs(q - n)
            no_fires_arr[index] = np.asarray([n,q])

            areas_norm[index] = normalize_percent(area_l_mean, area_k_mean)
            deviations_norm[index] = normalize_percent(r, o)
            no_fires_norm[index] = normalize_percent(q, n)

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
            
            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'val_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    area_mean = areas.mean()
    no_fires_mean = no_fires.mean()
    deviation_mean = deviations.mean()
    area_norm_mean = areas_norm.mean()
    no_fires_norm_mean = no_fires_norm.mean()
    deviation_norm_mean = deviations_norm.mean()
    no_fires_arr_mean = np.asarray([no_fires_arr[0].mean(),no_fires_arr[1].mean()])


    return mean_IoU, IoU_array, pixel_acc, mean_acc,area_mean,no_fires_mean,deviation_mean,area_norm_mean,no_fires_norm_mean,deviation_norm_mean,no_fires_arr_mean#,cosine_sim_mean


def test(config, test_dataset, testloader, model,
         sv_dir, sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.single_scale_inference(
                config,
                model,
                image.cuda())

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                
            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
