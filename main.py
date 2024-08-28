from tps_preprocessor import *
import cv2
import torch
import numpy as np

img = cv2.imread('img.jpg')
img = cv2.resize(img, (256,64), interpolation=cv2.INTER_AREA)
img_array = np.array(img)
img_tensor = torch.tensor(img_array)
img_tensor = img_tensor.unsqueeze(0)
img_tensor = img_tensor.permute(0, 3, 1, 2).to(torch.float32)

img_tensor = torch.randn(1,3,64,256)
print(img_tensor.shape)

preprocessor = STN(in_channels=3)
output = preprocessor(img_tensor)

print(output.shape)

# from datasets import TNGODataset

# dataset = TNGODataset('/data/CUBOX_VN_Recog_v7/CUBOX_VN_annotation_cleaned.json')
# print(dataset[300])

# print(len(dataset))