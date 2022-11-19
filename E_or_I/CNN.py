import torch
import torch.nn as nn # 신경망들이 포함됨
import torch.optim as optim # 최적화 알고리즘들이 포함힘
import torch.nn.init as init # 텐서에 초기값을 줌

import torchvision.datasets as datasets # 이미지 데이터셋 집합체
import torchvision.transforms as transforms # 이미지 변환 툴

from torch.utils.data import DataLoader # 학습 및 배치로 모델에 넣어주기 위한 툴

import numpy as np
import matplotlib.pyplot as plt
import cv2

class CNN(nn.Module):
    def __init__(self):
    	# super함수는 CNN class의 부모 class인 nn.Module을 초기화
        super(CNN, self).__init__()
        
        # batch_size = 100
        self.layer = nn.Sequential(
            # [100,1,28,28] -> [100,16,24,24]
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5),
            nn.ReLU(),
            
            # [100,16,24,24] -> [100,32,20,20]
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5),
            nn.ReLU(),
            
            # [100,32,20,20] -> [100,32,10,10]
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            # [100,32,10,10] -> [100,64,6,6]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(),
            
            # [100,64,6,6] -> [100,64,3,3]
            nn.MaxPool2d(kernel_size=2,stride=2)          
        )
        self.fc_layer = nn.Sequential(
        	# [100,64*3*3] -> [100,100]
            nn.Linear(64*3*3,100),                                              
            nn.ReLU(),
            # [100,100] -> [100,10]
            nn.Linear(100,10)                                                   
        )       
        
    def forward(self,x,batch_size=100):
    	# self.layer에 정의한 연산 수행
        out = self.layer(x)
        # view 함수를 이용해 텐서의 형태를 [100,나머지]로 변환
        out = out.view(batch_size,-1)
        # self.fc_layer 정의한 연산 수행    
        out = self.fc_layer(out)
        return out
batch_size=100
learning_rate=0.0001
num_epoch=20

trans=transforms.Compose([transforms.Resize((100,100)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train=datasets.ImageFolder(root="/home/kimyongtae/yolov5/mbti/train",transform=trans)


test=datasets.ImageFolder(root="/home/kimyongtae/yolov5/mbti/test",transform=trans)

classes=train.classes
print(classes)


train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_arr =[]
for i in range(num_epoch):
    for j,[image,label] in enumerate(train_loader):
        x = image.to(device)
        y= label.to(device)
        
        optimizer.zero_grad()
        
        output = model.forward(x)
        
        loss = loss_func(output,y)
        loss.backward()
        optimizer.step()
        
        
        if j % 1000 == 0:
            print(loss)
            loss_arr.append(loss.cpu().detach().numpy())

correct = 0
total = 0


# evaluate model
model.eval()

with torch.no_grad():
    for image,label in test_loader:
        x = image.to(device)
        y= label.to(device)

        output = model.forward(x)
        
        # torch.max함수는 (최댓값,index)를 반환 
        _,output_index = torch.max(output,1)
        
        # 전체 개수 += 라벨의 개수
        total += label.size(0)
        
        # 도출한 모델의 index와 라벨이 일치하면 correct에 개수 추가
        correct += (output_index == y).sum().float()
    
    # 정확도 도출
    print("Accuracy of Test Data: {}%".format(100*correct/total))
