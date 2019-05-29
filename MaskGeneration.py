#####
#Generate the mask image
####
import cv2
import json
import os
import numpy as np

def img_dir(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == ".jpg"  or os.path.splitext(file)[1] == ".jpeg":
                L.append(os.path.join(root,file))
    return L

def json_dir(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == ".json":
                L.append(os.path.join(root,file))
    return L

root = "painting_data/"
img_names = img_dir(root)
json_names = json_dir(root)
print(("json number is %d")%len(json_names))
index = 0
for json_name in json_names:
    print("current index %d"%index)
    load_f = open(json_name,"r")
    load_dict = json.load(load_f)

    img_addr = root + load_dict["imagePath"]
    img = cv2.imread(img_addr)
    gt_img = np.zeros(img.shape[0:2], np.uint8)
    contour_points = []
    print(("shape length %d")%len(load_dict["shapes"]))
    if len(load_dict["shapes"]) == 0:
        print("null shapes")
        index = index + 1
        continue

    for point in load_dict["shapes"][0]["points"]:
        contour_points.append(point)
    ###the label mask made from json, should change with the json format
    label = load_dict['~~']['~~']
    ########################
    contour_points = np.array(contour_points)
    cv2.fillConvexPoly(gt_img, contour_points, 255)
    cv2.imwrite("data/" + str(index) + ".jpg", img)
    cv2.imwrite("mask/" + str(index) + ".png", gt_img)
    index = index + 1

load_f.close()
print("over!")

