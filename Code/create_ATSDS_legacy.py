import torch

from collections import defaultdict 
import numpy as np
import pandas as pd
import os
import cv2
import torchvision
import torchvision.datasets as dataset
import random
from PIL import Image, ImageDraw

random.seed(42)
np.random.seed(42)

#BASE_DIR = "/data/horse/ws/juul507e-ju_streetsigns/data/"
BASE_DIR = "data/"

trainset = dataset.GTSRB(
    root=BASE_DIR, split="train", download=False)
testset = dataset.GTSRB(
    root=BASE_DIR, split="test", download=False)


round_classes = [1,2,3,4,5,7,8,9,10,17,35,38]
triangel_class = [11,18,25,31] 
triangel_class_reverse = [13]
stop_sign_class = [14]
rhombus_class=[12]


def transform_numpy_array(x):
    return np.array(x)
    
def unit_circle_filled(r):
    A = np.arange(-r,r)**2
    dists = np.sqrt(A[:,None] + A)
    return np.moveaxis(np.tile((np.abs(dists)<r).astype(int),(3,1,1)),0,2)

def get_cutout_mask(target_class,resize_size):
    if int(target_class) in round_classes:
        A = np.arange(-(resize_size//2),resize_size//2)**2
        dists = np.sqrt(A[:,None] + A)
        return np.moveaxis(np.tile((np.abs(dists)<(resize_size//2)).astype(int),(3,1,1)),0,2)
    elif int(target_class) in triangel_class:
        mask = Image.new('L',(resize_size,resize_size))
        ImageDraw.Draw(mask).polygon([(0,resize_size),(resize_size//2,0),(resize_size,resize_size),(0,resize_size)],outline=1,fill=1)
        return np.moveaxis(np.tile(np.array(mask).astype(int),(3,1,1)),0,2)
    elif int(target_class) in triangel_class_reverse:
        mask = Image.new('L',(resize_size,resize_size))
        ImageDraw.Draw(mask).polygon([(0,0),(resize_size,0),(resize_size//2,resize_size),(0,0)],outline=1,fill=1)
        return np.moveaxis(np.tile(np.array(mask).astype(int),(3,1,1)),0,2)
    elif int(target_class) in rhombus_class:
        mask = Image.new('L',(resize_size,resize_size))
        ImageDraw.Draw(mask).polygon([(resize_size//2,resize_size),(0,resize_size//2),(resize_size//2,0),(resize_size,resize_size//2)],outline=1,fill=1)
        return np.moveaxis(np.tile(np.array(mask).astype(int),(3,1,1)),0,2)
    elif int(target_class) in stop_sign_class:
        mask = Image.new('L',(resize_size,resize_size))
        ImageDraw.Draw(mask).polygon([(resize_size//3,resize_size),(0,resize_size//1.5),(0,resize_size//3),(resize_size//3,0),(resize_size//1.5,0),(resize_size,resize_size//3),(resize_size,resize_size//1.5),(resize_size//1.5,resize_size),(resize_size//3,resize_size)],outline=1,fill=1)
        return np.moveaxis(np.tile(np.array(mask).astype(int),(3,1,1)),0,2)
        
#reverse_triangle_class = [13]
labelcount = defaultdict(int)
#classfolders = ['00002', '00007', '00009', '00010', '00011', '00012', '00018', '00025', '00035', '00038']
classfolders = []
for i in trainset:
    labelcount[str(i[1])] +=1
for amount,label in zip(labelcount.values(),labelcount.keys()):
    if (amount>500):
        classfolders.append(str(label).zfill(5))
print(classfolders)
object_areas = {}
for dir in classfolders:
    locations = pd.read_csv(BASE_DIR + "gtsrb/GTSRB/Training/" + dir + "/GT-" + dir +".csv")
    object_areas[dir] = locations
    
#df_back = pd.DataFrame(backgroundimageindex)
#df_back.to_csv(BASE_DIR + "backgroundindice.csv")
objectimageindex= random.sample(range(500), 500)
backgroundindex= random.sample(range(len(classfolders)*(500+50)), len(classfolders)*(500+50))


objectimageindex_train = objectimageindex[0:450]
objectimageindex_test = objectimageindex[450:]
backgroundindex_train = backgroundindex[0:len(classfolders)*500]
backgroundindex_test = backgroundindex[len(classfolders)*500:]

#df_obj = pd.DataFrame(objectimageindex)
#df_obj.to_csv(BASE_DIR + "objectindice.csv")

RESIZE_SIZE = 128
BACKGROUNDS_PER_SIGN = 20

counter_train = 0
counter_test = 0
labelfile = [["Filename","Roi.X1","Roi.Y1","Roi.X2","Roi.Y2","ClassId"]]
for classname in classfolders:
    if not os.path.isdir(BASE_DIR + "atsds_large/train/" + classname):
        os.makedirs(BASE_DIR + "atsds_large/train/" + classname)
    if not os.path.isdir(BASE_DIR + "atsds_large_mask/train/" + classname):
        os.makedirs(BASE_DIR + "atsds_large_mask/train/" + classname)
    if not os.path.isdir(BASE_DIR + "atsds_large_background/train/" + classname):
        os.makedirs(BASE_DIR + "atsds_large_background/train/" + classname)
        
    for j in objectimageindex_train:
        background = np.array(Image.open(BASE_DIR + "streetimages/" + str(backgroundindex_train[counter_train]).zfill(6) + ".png"))
        background = cv2.resize(np.copy(background),dsize=(512,512))
        backgroundcopy = np.copy(background)
        backgroundsave = Image.fromarray(backgroundcopy)
        backgroundsave.save(BASE_DIR + "atsds_large_background/train/" + classname + "/" +str(counter_train).zfill(6) + ".png")

        object = np.array(object_areas[classname])[j][0].split(';')
        im_obj = np.array(Image.open(BASE_DIR + "gtsrb/GTSRB/Training/" + classname + "/" + object[0]))
        obj_sx = int(object[3])
        obj_sy = int(object[4])
        obj_endx = int(object[5])
        obj_endy = int(object[6])
        obj_cutout = np.copy(im_obj[obj_sy:obj_endy,obj_sx:obj_endx])
        resize_size = RESIZE_SIZE
        obj_cutout = cv2.resize(obj_cutout,dsize=(resize_size,resize_size))
        sx = np.random.randint(background.shape[1]-obj_cutout.shape[1])
        sy = np.random.randint(background.shape[0]-obj_cutout.shape[0])
        cutout_mask = get_cutout_mask(classname,resize_size)
        insertionarea = np.copy(backgroundcopy[sy:sy+obj_cutout.shape[1],sx:sx+obj_cutout.shape[0]])
        insertionarea = np.where(cutout_mask != 1,  insertionarea, obj_cutout)
        backgroundcopy[sy:sy+obj_cutout.shape[1],sx:sx+obj_cutout.shape[0]] = insertionarea
        dataimg = Image.fromarray(backgroundcopy)
        dataimg.save(BASE_DIR + "atsds_large/train/" + classname + "/" +str(counter_train).zfill(6) + ".png")

        mask_backgroud = np.zeros_like(background)
        mask_insertionarea = np.where(cutout_mask != 1,  0, 255)
        mask_backgroud[sy:sy+obj_cutout.shape[1],sx:sx+obj_cutout.shape[0]] = mask_insertionarea
        maskimg = Image.fromarray(mask_backgroud)
        maskimg.save(BASE_DIR + "atsds_large_mask/train/" + classname + "/" +str(counter_train).zfill(6) + ".png") 
        labelfile.append([str(counter_train).zfill(6) + ".png",sx,sy,sx+obj_cutout.shape[0],sy+obj_cutout.shape[1],object[-1]])
        counter_train+=1
    if not os.path.isdir(BASE_DIR + "atsds_large/test/" + classname):
        os.makedirs(BASE_DIR + "atsds_large/test/" + classname)
    if not os.path.isdir(BASE_DIR + "atsds_large_mask/test/" + classname):
        os.makedirs(BASE_DIR + "atsds_large_mask/test/" + classname)
    if not os.path.isdir(BASE_DIR + "atsds_large_background/test/" + classname):
        os.makedirs(BASE_DIR + "atsds_large_background/test/" + classname)

    for j in objectimageindex_test:
        background = np.array(Image.open(BASE_DIR + "streetimages/" + str(backgroundindex_test[counter_test]).zfill(6) + ".png"))
        background = cv2.resize(np.copy(background),dsize=(512,512))
        backgroundcopy = np.copy(background)
        backgroundsave = Image.fromarray(backgroundcopy)
        backgroundsave.save(BASE_DIR + "atsds_large_background/test/" + classname + "/" +str(counter_test).zfill(6) + ".png")
        object = np.array(object_areas[classname])[j][0].split(';')
        im_obj = np.array(Image.open(BASE_DIR + "gtsrb/GTSRB/Training/" + classname + "/" + object[0]))
        obj_sx = int(object[3])
        obj_sy = int(object[4])
        obj_endx = int(object[5])
        obj_endy = int(object[6])
        obj_cutout = np.copy(im_obj[obj_sy:obj_endy,obj_sx:obj_endx])
        resize_size = RESIZE_SIZE
        obj_cutout = cv2.resize(obj_cutout,dsize=(resize_size,resize_size))
        sx = np.random.randint(background.shape[1]-obj_cutout.shape[1])
        sy = np.random.randint(background.shape[0]-obj_cutout.shape[0])
        cutout_mask = get_cutout_mask(classname,resize_size)
        insertionarea = np.copy(backgroundcopy[sy:sy+obj_cutout.shape[1],sx:sx+obj_cutout.shape[0]])
        insertionarea = np.where(cutout_mask != 1,  insertionarea, obj_cutout)
        mask_insertionarea = np.where(cutout_mask != 1,  0, 255)
        mask_backgroud = np.zeros_like(backgroundcopy)
        backgroundcopy[sy:sy+obj_cutout.shape[1],sx:sx+obj_cutout.shape[0]] = insertionarea
        mask_backgroud [sy:sy+obj_cutout.shape[1],sx:sx+obj_cutout.shape[0]] = mask_insertionarea 
        maskimg = Image.fromarray(mask_backgroud)
        dataimg = Image.fromarray(backgroundcopy)
        dataimg.save(BASE_DIR + "atsds_large/test/" + classname + "/" +str(counter_test).zfill(6) + ".png")

        maskimg.save(BASE_DIR + "atsds_large_mask/test/" + classname + "/" +str(counter_test).zfill(6) + ".png") 
        labelfile.append([str(counter_test).zfill(6) + ".png",sx,sy,sx+obj_cutout.shape[0],sy+obj_cutout.shape[1],object[-1]])
        counter_test+=1

df_labelfile = pd.DataFrame(labelfile)
df_labelfile.to_csv(BASE_DIR + "labelfile.csv")            
