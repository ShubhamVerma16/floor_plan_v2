import os
import cv2
import shutil
import numpy as np


def mkdirs(dest_dir):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    dest_img = dest_dir + "images/"
    dest_labels = dest_dir + "labels/"

    img_train = dest_img + "train/"
    img_val = dest_img + "val/"

    label_train = dest_labels + "train/"
    label_val = dest_labels + "val/"

    if not os.path.exists(dest_img):
        os.mkdir(dest_img)

    if not os.path.exists(dest_labels):
        os.mkdir(dest_labels)

    if not os.path.exists(img_train):
        os.mkdir(img_train)

    if not os.path.exists(img_val):
        os.mkdir(img_val)

    if not os.path.exists(label_train):
        os.mkdir(label_train)

    if not os.path.exists(label_val):
        os.mkdir(label_val)



def gen_mmdata(src_dir, dest_dir):
    mkdirs(dest_dir)

    for folder in os.listdir(src_dir):
        for subfold in os.listdir(src_dir + folder):
            for image_nm in os.listdir(src_dir + folder + "/" + subfold + "/image/"):
                src_path = src_dir + folder + "/" + subfold + "/image/" + image_nm
                dst_path = dest_dir + "images/" + folder + "/" + subfold + "_" + image_nm
                print("Writing image: ", image_nm)
                shutil.copy(src_path, dst_path)
                #tf.write(image_nm + _leftImg8bit.png)

            for image_nm in os.listdir(src_dir + folder + "/" + subfold + "/gT/"):
                src_path = src_dir + folder + "/" + subfold + "/gT/" + image_nm
                dst_path = dest_dir + "labels/" + folder + "/" + subfold + "_" + image_nm
                print("Writing image: ", image_nm)
                img = cv2.imread(src_path)
                print("Unique pixels: ", np.unique(img.flatten()))
                shutil.copy(src_path, dst_path)



dest_folder = "./processed/mmseg/"
src_folder = "./processed/esanet/"
train_txt = "./processed/mmseg/train.txt"
val_txt = "./processed/mmseg/val.txt"
gen_mmdata(src_folder, dest_folder)