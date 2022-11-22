import torch
import cv2
model=torch.load('/home/kimyongtae/yolov5/ckpt_epoch_299.pth')
img=cv2.imread('/home/kimyongtae/yolov5/test/images3.jpeg')

print(model)