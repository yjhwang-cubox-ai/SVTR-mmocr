from tps_preprocessor import *
from svtr_encoder import *
import cv2
import torch
import numpy as np

img = cv2.imread('img.jpg')
img = cv2.resize(img, (256,64), interpolation=cv2.INTER_AREA)
img_array = np.array(img)
img_tensor = torch.tensor(img_array)
img_tensor = img_tensor.unsqueeze(0)
img_tensor = img_tensor.permute(0, 3, 1, 2).to(torch.float32)

img_tensor = torch.randn(2,3,64,256)
print(f'Data Shape: {img_tensor.shape}')

preprocessor = STN(in_channels=3)
encoder = SVTREncode()
output = preprocessor(img_tensor)

print(f'STN output: {output.shape}')

output = encoder(output)

print(f'SVTREncode output: {output.shape}')