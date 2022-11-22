import os
import numpy as np
import matplotlib.pyplot as plt
# folder_path="/home/kimyongtae/yolov5/runs/detect/exp9/labels"
# def get_files_count(folder_path):
# 	dirListing = os.listdir(folder_path)
# 	return len(dirListing)
	
# if __name__ == "__main__":
# 	print(get_files_count(folder_path))
def mbti(path):
 folder_path=path
 dirListing = os.listdir(folder_path)
 count_person=[]
 for i in dirListing:
  file=open(folder_path+i)
  lines = file.readlines()
  count_num=0
  for k in range(len(lines)):
      if lines[k][0:2]=="0 ":
          count_num+=1
      
  count_person.append(count_num)               
  file.close()

#  while True:
#     c = file.read()
#     data_frame=i+' '+c
#     if c == '':
#         break
#     print(data_frame, end='')
 Count_mbti_E=[]
 Count_mbti_I=[]
 count_mbti_ALL=[]
 for i in count_person:
    count_mbti_ALL.append(1)
    if i>=4:
        Count_mbti_E.append(1)
        Count_mbti_I.append(0)
    elif i==0:
        Count_mbti_E.append(0.25)
        Count_mbti_I.append(0.75)
    elif i==1:
        Count_mbti_E.append(0.4)
        Count_mbti_I.append(0.6)
    elif i==2:
        Count_mbti_E.append(0.6)
        Count_mbti_I.append(0.4)
    elif i==3:
        Count_mbti_E.append(0.9)
        Count_mbti_I.append(0.1)

 count_E=np.array(Count_mbti_E)
 count_I=np.array(Count_mbti_I)
 count_ALL=np.array(count_mbti_ALL)
 All=np.sum(count_ALL)
 E=np.sum(count_E)
 I=np.sum(count_I)
 E_percent=E/All
 I_percent=I/All
 print('person is E: ',(E_percent)*100,'%')
 print('person is I: ',(I_percent)*100,'%')

 labels=['E','I']
 sizes=[E_percent,I_percent]
 explode=(0.1,0.1)

 fig1,ax1=plt.subplots()
 ax1.pie(sizes,explode=explode, labels=labels, autopct='%1.1f%%',shadow=True,startangle=90)
 ax1.axis('equal')
 plt.savefig('E vs I.png')

mbti('/home/kimyongtae/yolov5/result/mbti2/labels/')

