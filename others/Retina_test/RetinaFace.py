import natsort
import os
from PIL import Image

from retinaface import RetinaFace
import matplotlib.pyplot as plt

#https://github.com/serengil/retinaface
# Test Code for RetinaFace 

#idx= 114 # target index

'''root = "/data2/MS-FaceSynthetic"
img_path = os.path.join(root, "img")
img_list = natsort.natsorted(os.listdir(img_path))
num_of_sample = 2'''
#for idx in range(num_of_sample):
img_path = "/root/landmark_detection/test_image/000000.png"
img = Image.open(img_path)

resp = RetinaFace.detect_faces(img_path = img_path)
print(resp)
print("resp['face_1']['facial_area']: ", resp["face_1"]['facial_area'])
print("resp['face_1']['facial_area'][0]: ", resp["face_1"]['facial_area'][0])
print("resp['face_1']['facial_area'][1]: ", resp["face_1"]['facial_area'][1])
print("resp['face_1']['facial_area'][2]: ", resp["face_1"]['facial_area'][2])
print("resp['face_1']['facial_area'][3]: " ,resp["face_1"]['facial_area'][3])

print("2 - 0 ", resp["face_1"]['facial_area'][2] - resp["face_1"]['facial_area'][0])
print("3 - 1 ", resp["face_1"]['facial_area'][3] - resp["face_1"]['facial_area'][1])

print("type: resp['face_1']['facial_area']: ", type(resp["face_1"]['facial_area'][1]))

faces = RetinaFace.extract_faces(img_path = img_path, align = False)
plt.imshow(img)
plt.savefig('RetinaFace_GT_%d.png'%(0))

plt.clf()
plt.imshow(faces[0])
plt.savefig('RetinaFace_test_%d.png'%(0))