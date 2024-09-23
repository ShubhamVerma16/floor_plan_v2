import os
import shutil


input_folder = "./processed/mmseg/"

image_suffix = '_leftImg8bit.png'
label_suffix = '_gtFine_labelTrainIds.png'

dest_folder = "./processed/mmseg_point_rend/"
train_txt = dest_folder + "train.txt"
val_txt = dest_folder + "val.txt"

for folder in os.listdir(input_folder):
    suffix = image_suffix if folder == "images" else label_suffix
    for subdir in os.listdir(os.path.join(input_folder, folder)):
        txt_file = train_txt if subdir == "train" else val_txt
        train_val_pth = os.path.join(input_folder, folder, subdir)
        with open(txt_file, 'a') as txt_file:
            for img_nm in os.listdir(train_val_pth):
                img_pth = os.path.join(train_val_pth, img_nm)
                dest_pth = os.path.join(dest_folder, folder, subdir, img_nm.split('.')[0] + suffix)
                shutil.copy(img_pth, dest_pth)
                txt_file.write(img_nm.split('.')[0]+"\n")
