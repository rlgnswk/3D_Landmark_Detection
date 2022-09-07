import natsort
import os
from PIL import Image

from retinaface import RetinaFace
import matplotlib.pyplot as plt

#https://github.com/serengil/retinaface
# Test Code for RetinaFace 

idx= 114 # target index

root = "/data2/MS-FaceSynthetic"
img_path = os.path.join(root, "img")
img_list = natsort.natsorted(os.listdir(img_path))

#for idx in range(10):
img = Image.open(img_path + '/' + img_list[idx])

resp = RetinaFace.detect_faces(img_path = img_path + '/' + img_list[idx])
print(resp)
print(resp["face_1"]['facial_area'])
print(resp["face_1"]['facial_area'][:2])
print(resp["face_1"]['facial_area'][0])
print(resp["face_1"]['facial_area'][1])
print(resp["face_1"]['facial_area'][2])
print(resp["face_1"]['facial_area'][3])
print(type(resp["face_1"]['facial_area'][1]))
'''faces = RetinaFace.extract_faces(img_path = img_path + '/' + img_list[idx], align = False)
plt.imshow(img)
plt.savefig('RetinaFace_GT_%d.png'%(idx))

plt.clf()

plt.imshow(faces[0])
plt.savefig('RetinaFace_test_%d.png'%(idx))'''