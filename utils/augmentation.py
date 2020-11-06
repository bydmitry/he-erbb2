import random
from io import BytesIO

import numpy as np
import pandas as pd

import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

def augment_random_crop(img, crop_size):
    h, w = img.shape[0], img.shape[1]

    pad_h, pad_w = 0, 0
    if w < crop_size:
        pad_w = np.uint16(np.ceil((crop_size - w)/2) + int(crop_size*.1))
    if h < crop_size:
        pad_h = np.uint16(np.ceil((crop_size - h)/2) + int(crop_size*.1))
    if  pad_w or pad_h:
        img   = img[:]
        img   = cv2.copyMakeBorder(img,pad_h,pad_h,pad_w,pad_w,cv2.BORDER_CONSTANT,value=0)
        h, w  = img.shape[0], img.shape[1]

    x1 = random.randint(0, w - crop_size)
    y1 = random.randint(0, h - crop_size)
    return img[y1:y1+crop_size, x1:x1+crop_size,:]


def center_crop(img, crop_size, fill=[0,0,0]):
    h, w = img.shape[0], img.shape[1]

    pad_h, pad_w = 0, 0
    if w < crop_size:
        pad_w = np.uint16(np.ceil((crop_size - w)/2))
    if h < crop_size:
        pad_h = np.uint16(np.ceil((crop_size - h)/2))
    if  pad_w or pad_h:
        img   = img[:]
        img   = cv2.copyMakeBorder(img,pad_h,pad_h,pad_w,pad_w,cv2.BORDER_CONSTANT,value=fill)
        h, w  = img.shape[0], img.shape[1]

    x1 = w//2-crop_size//2
    y1 = h//2-crop_size//2
    return img[y1:y1+crop_size, x1:x1+crop_size,:]

def five_crop(img, crop_size):
    h, w = img.shape[0], img.shape[1]
    c_cr = center_crop(img, crop_size)

    pad_h, pad_w = 0, 0
    if w < crop_size:
        pad_w = np.uint16(np.ceil((crop_size - w)/2))
    if h < crop_size:
        pad_h = np.uint16(np.ceil((crop_size - h)/2))
    if  pad_w or pad_h:
        img   = img[:]
        img   = cv2.copyMakeBorder(img,pad_h,pad_h,pad_w,pad_w,cv2.BORDER_CONSTANT,value=0)
        h, w  = img.shape[0], img.shape[1]

    ul_cr = img[0:crop_size, 0:crop_size]      # upper-left crop
    ur_cr = img[0:crop_size, w-crop_size:w]    # upper-right crop
    bl_cr = img[h-crop_size:h, 0:crop_size]    # bottom-left crop
    br_cr = img[h-crop_size:h, w-crop_size:w]  # bottom-right crop

    return np.stack((c_cr, ul_cr, ur_cr, bl_cr, br_cr))

def cross_five_crop(img, crop_size):
    h, w = img.shape[0], img.shape[1]
    c_cr = center_crop(img, crop_size)

    pad_h, pad_w = 0, 0
    if w < crop_size:
        pad_w = np.uint16(np.ceil((crop_size - w)/2))
    if h < crop_size:
        pad_h = np.uint16(np.ceil((crop_size - h)/2))
    if  pad_w or pad_h:
        img   = img[:]
        img   = cv2.copyMakeBorder(img,pad_h,pad_h,pad_w,pad_w,cv2.BORDER_CONSTANT,value=0)
        h, w  = img.shape[0], img.shape[1]

    xx = w//2-crop_size//2
    yy = h//2-crop_size//2

    l_cr = img[yy:yy+crop_size, 0:crop_size]
    r_cr = img[yy:yy+crop_size, w-crop_size:w]
    t_cr = img[0:crop_size, xx:xx+crop_size]
    b_cr = img[h-crop_size:h, xx:xx+crop_size]

    return np.stack((c_cr, l_cr, r_cr, t_cr, b_cr))

def flips(img):
    stck = np.stack((img, cv2.flip(img, 0), cv2.flip(img, 1)))
    return stck

def augment_random_flip(img, hprob=0.5, vprob=0.5):
    img = img.copy()
    if random.random() > hprob:
        img = cv2.flip(img, 1)

    if random.random() > vprob:
        img = cv2.flip(img, 0)

    return img

def adjust_gamma(img, gamma=1.0):
    return np.uint8(cv2.pow(img / 255., gamma)*255.)

def augment_random_gamma(img, gammas):
    #gamma = random.uniform(*gammas)
    gamma = gammas
    return adjust_gamma(img, gamma)

def augment_random_linear(img, sr=5, ssx=0.1, ssy=0.1, inter=cv2.INTER_LINEAR):

    img = img.copy()

    rot = (np.random.rand(1)[0]*2-1)*sr
    scalex = np.random.rand(1)[0]*ssx
    scaley = np.random.rand(1)[0]*ssy

    R = np.array([np.cos(np.deg2rad(rot)), np.sin(np.deg2rad(rot)), 0,
                  -np.sin(np.deg2rad(rot)), np.cos(np.deg2rad(rot)), 0,
                  0, 0, 1
                 ]).reshape((3,3))

    S = np.array([1, scalex, 0,
                  scaley, 1, 0,
                 0, 0, 1]).reshape((3,3))


    A = np.dot(R, S)

    return cv2.warpAffine(img, A.T[:2, :], img.shape[1::-1], inter, borderMode=cv2.BORDER_REFLECT)


def augment_scale(img, scale, inter=cv2.INTER_LINEAR):
    img = img.copy()
    w, h = img.shape[1::-1]
    scale_factor = scale
    img = cv2.resize(img, (int(w*scale_factor), int(h*scale_factor)), interpolation=inter)

    return img

def augment_random_scale(img, scales, inter=cv2.INTER_LINEAR):
    img = img.copy()
    w, h = img.shape[1::-1]
    scale_factor = random.uniform(*scales)
    img = cv2.resize(img, (int(w*scale_factor), int(h*scale_factor)), interpolation=inter)

    return img

def rotate_90(img):
    res = np.rot90(img, random.choice(range(4)), (0,1)).copy()
    return res