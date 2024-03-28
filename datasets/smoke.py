# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image

import torch
from .base_dataset import BaseDataset

def maximum_filter(n,img):
    size = (n,n)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape,size)

    img_result = cv2.dilate(img,kernel)
    img_result = cv2.erode(img_result,kernel)
    return img_result


class Smoke(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_classes=2,
                 multi_scale=False,
                 flip=True,
                 ignore_label=255,
                 base_size=1024,
                 crop_size=(1024, 1024),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4,
                 channel_type='rgb'):

        super(Smoke, self).__init__(ignore_label, base_size,
                                     crop_size, scale_factor, mean, std,channel_type)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip

        self.img_list = [line.strip().split() for line in open(root + list_path)]

        self.files = self.read_files()

        self.label_mapping = {-1: ignore_label, 0: ignore_label,1: 1}
        self.color_list = [[0, 0, 0],[1,1,1]]
        self.class_weights = None #torch.FloatTensor([0.5034,74.5384]).cuda()

        self.bd_dilate_size = bd_dilate_size

    def read_files(self):
        files = []

        for item in self.img_list:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "img": image_path,
                "label": label_path,
                "name": name
            })

        return files

    def color2label(self, color_map):
        label = np.ones(color_map.shape[:2]) * self.ignore_label
        for i, v in enumerate(self.color_list):
            label[(color_map == v).sum(2) == 3] = i

        return label.astype(np.uint8)

    def label2color(self, label):
        color_map = np.zeros(label.shape + (3,))
        for i, v in enumerate(self.color_list):
            color_map[label == i] = self.color_list[i]

        return color_map.astype(np.uint8)

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = Image.open(item["img"]).convert('RGB')
        image = np.array(image)
        #image = cv2.resize(image,(1024,1024))



        color_map = Image.open( item["label"]).convert('RGB')
        color_map = np.array(color_map)
        label = self.color2label(color_map)
        #label = cv2.resize(label,(1024,1024))

        #hsv = cv2.cvtColor(cv2.imread(item["img"]), cv2.COLOR_BGR2HSV)

        #mask = hsv[:, :, 1]
        #_, thr = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)
        #bb = cv2.medianBlur(thr, 5)
        #zz = cv2.cvtColor(bb, cv2.COLOR_GRAY2BGR)
        #mask_contours_enhanced = maximum_filter(5, zz)

        #image = np.concatenate((image, mask_contours_enhanced), axis=2)

        #image = np.concatenate((image, hsv), axis=2)

        #image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        size = image.shape
        image, label, edge = self.gen_sample(image, label,
                                             self.multi_scale, self.flip, edge_pad=False,
                                             edge_size=self.bd_dilate_size, city=False)

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred


    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.label2color(preds[i])
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

