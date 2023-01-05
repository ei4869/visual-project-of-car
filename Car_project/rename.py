import os
path = 'Stop/'
fileList=os.listdir(path)
n=0
for i in fileList:
    
    #设置旧文件名（就是路径+文件名）
    oldname=path+ os.sep + fileList[n]   # os.sep添加系统分隔符
    
    #设置新文件名
    newname=path + os.sep+'img2-'+str(n+1)+'.jpg'
    try:
        os.rename(oldname,newname)   #用os模块中的rename方法对文件改名
        print(oldname,'======>',newname)
    except Exception:
        pass
    
    n+=1
