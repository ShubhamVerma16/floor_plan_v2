import os
import cv2
import json
import shutil
import numpy as np
#from numba import njit, jit, cuda
#from numba.typed import Dict

pixel_map = {"green_indoor": 1, #red - combo
            "blue_outdoor": 2, #blue - outdoor
            "red_enclose": 3, #green - indoor
            }

def processJson(json_path):
    rooms = {}
    with open(json_path, 'r') as annot:
        data = json.load(annot)
        for key, val in data['_via_img_metadata'].items():
            img_nm = val["filename"]
            rooms[img_nm] = {"green_indoor": [],
                          "blue_outdoor": [],
                          "red_enclose": []}
            #print(rooms)

            for region in val["regions"]:
                # if region["region_attributes"]["area_type"] == "red":
                #     rooms[img_nm]["red_enclose"].append([region["shape_attributes"]["all_points_x"], region["shape_attributes"]["all_points_y"]])
                # elif region["region_attributes"]["area_type"] == "blue":
                #     rooms[img_nm]["blue_outdoor"].append([region["shape_attributes"]["all_points_x"], region["shape_attributes"]["all_points_y"]])
                # elif region["region_attributes"]["area_type"] == "green":
                rooms[img_nm]["green_indoor"].append([region["shape_attributes"]["all_points_x"], region["shape_attributes"]["all_points_y"]])

    #with open(processed_json, 'a') as pf:
    #    pf.write(json.dumps(rooms, indent=4))

    #print(rooms)
    return rooms

    


def generateMask(mask_img, polygons):
    for region, bboxes in polygons.items():
        for all_x, all_y in bboxes:
            coords = [[x, y] for x, y in zip(all_x, all_y)]
            poly = np.array(coords, dtype=np.int32)
            poly = poly.reshape((-1, 1, 2))

            #for x, y in coords:
            #    mask_img[y][x] = 255

            if region == "green_indoor":
                #print(poly)
                #print("value: ", pixel_map["green_indoor"])
                cv2.fillPoly(mask_img, [poly], pixel_map["green_indoor"], lineType=cv2.LINE_8)

            if region == "blue_outdoor":
                cv2.fillPoly(mask_img, [poly], pixel_map["blue_outdoor"], lineType=cv2.LINE_8)

    print(mask_img.shape)
    print("Sum: ", cv2.countNonZero(mask_img))
    #sum_ = 0
    #for i in range(1000):
    #    for j in range(1000):
    #        if mask_img[i][j] != 0:
    #            sum_ += 1

    #print("Sum: ", sum_)
    #cv2.imwrite("image.png", mask_img)
    return mask_img 



def process_new_data(folder_path, dest_path, json_path):
    data = processJson(json_path)
    total_imgs = 50 #len(data)
    
    #copy_folder = "train/printed/"
    idx = 0 
    #total_imgs = len(os.listdir(input_path))
    for img_nm, polygons in data.items():
        print(img_nm)
        img_pth = input_path + img_nm
        
        try:
            #read image and initialize mask
            img_rgb = cv2.imread(img_pth)
            rows, cols, _ = img_rgb.shape
            mask_img = np.zeros([rows, cols, 1], dtype = np.uint8)
            
            #generate mask
            ground_truth = generateMask(mask_img, polygons)
            
            if idx >= int(0.80*total_imgs):
                dest_img_pth = val_path + "image/" + img_nm
                dest_gt_pth = val_path + "gT/" + img_nm
            else:
                dest_img_pth = train_path + "image/" + img_nm
                dest_gt_pth = train_path + "gT/" + img_nm

            if cv2.countNonZero(ground_truth):
                shutil.copy(img_pth, dest_img_pth)
                print("Writing gT: ", np.unique(ground_truth.flatten()))
                cv2.imwrite(dest_gt_pth, ground_truth)

            idx += 1

        except Exception as e:
            print(e)
            continue
    print("Total imgs: ", total_imgs)
    print("Train len: ", 0.80*total_imgs)


input_path = "./raw_data/images/200-299/"
json_path = "./raw_data/annots/154-254.json"

dest_path = "./processed/esanet/"
train_path = dest_path + "train/printed/"
val_path = dest_path + "val/printed/"

process_new_data(input_path, dest_path, json_path)
