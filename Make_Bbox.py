import natsort
import os
from PIL import Image

from retinaface import RetinaFace
import matplotlib.pyplot as plt

root = "/data2/MS-FaceSynthetic"
img_path = os.path.join(root, "img")
img_list = natsort.natsorted(os.listdir(img_path))

X_mean = 0 
Y_mean = 0 
num_of_None_detection = 0
print("total length: ", len(img_list))
bbox_leftcorner_coord_path = os.path.join(root, "bbox_leftcorner_coord") # per-defined floder

for idx in range(len(img_list)):
    print(idx, " ing ...")
    num_str = str(idx).zfill(6)
    file = open(bbox_leftcorner_coord_path + "/" + num_str +"_bbox.txt", "w") 

    resp = RetinaFace.detect_faces(img_path = img_path + '/' + img_list[idx])
    #print(resp["face_1"]['facial_area']) # [201, 158, 383, 407] -> x1, y1 , x2, y2
    
    #resp["face_1"]['facial_area'][:2][0] # y coordinate left corner
    #resp["face_1"]['facial_area'][:2][1] # x coordinate left corner   
    
    try:
        #check whether it is detected
        X_mean = resp["face_1"]['facial_area'][2] - resp["face_1"]['facial_area'][0]

        #X_extra = 128 - resp["face_1"]['facial_area'][:2][0]
        
        if resp["face_1"]['facial_area'][0] + 256 >= 512:
            resp["face_1"]['facial_area'][0] = 255
        if resp["face_1"]['facial_area'][1] + 256 >= 512:
            resp["face_1"]['facial_area'][1] = 255
        
        #img = Image.open(img_path + '/' + img_list[idx])
        #plt.imshow(img.crop((resp["face_1"]['facial_area'][0], resp["face_1"]['facial_area'][1], resp["face_1"]['facial_area'][0]+ 256, resp["face_1"]['facial_area'][1] + 256)))
        #plt.savefig('bbox_image_here.png')

        #print("????")
        #exit(0)
        X_extra = (256 - (resp["face_1"]['facial_area'][2] - resp["face_1"]['facial_area'][0]))//2

        file.write(str(resp["face_1"]['facial_area'][0]- X_extra))
        file.write(" ")
        file.write(str(resp["face_1"]['facial_area'][1]))

    except TypeError:
        num_of_None_detection += 1
        
        #assume centered box 
        file.write("128")
        file.write(" ")
        file.write("128")
        pass

    file.close()

print("num_of_None_detection: ", num_of_None_detection)
print("Done")