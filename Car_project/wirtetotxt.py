
# -*- coding: utf-8 -*-
# 生成文件夹中所有文件的路径到txt
import os
# def listdir(path, list_name):  # 传入存储的list
#     for file in os.listdir(path):
#         file_path = os.path.join(path, file)
#         if os.path.isdir(file_path):
#             listdir(file_path, list_name)
#         else:
#             list_name.append(file_path)
 
# list_name=[]
path='C:/Users/4869/Desktop/Car_project/data/val/'   #文件夹路径
# listdir(path,list_name)
# print(list_name)
 
# with open('train.txt','w') as f:     #要存入的txt
#     write=''
#     for i in list_name:
#         write=write+str(i)+'\n'
#     f.write(write)

with open('C:/Users/4869/Desktop/Car_project/data/val.txt','a') as f:     #要存入的txt
    
    for i in range(39,51):
        write=path+'img2-'+str(i)+'.jpg'+'\n'
        f.write(write)