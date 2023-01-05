import cv2
import numpy as np
import os
import time
#import serial


WIDTH = 640
HEIGHT = 480

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
#ser = serial.Serial("/dev/ttyAMA0", 115200)

PATH1 = "WaitStop/"
PATH2 = "Stop/"
SVMNAME = "test1"
TRAIN_NUM = 30
COLOR_DICT = {
                'highred':[np.array([156,43,46]),np.array([180,255,255])],
                'lowred':[np.array([0,43,46]),np.array([10,255,255])],
                'green':[np.array([35,43,46]),np.array([77,255,255])],
                'blue':[np.array([100,43,46]),np.array([124,255,255])],
                'yellow':[np.array([26,43,46]),np.array([34,255,255])],
                'orange':[np.array([11,43,46]),np.array([25,255,255])]}
COLORS = ['highred','lowred','green','blue','yellow','orange']

#1 识别等停和停止 SVM二值回归
class WaitandStop():
    def __init__(self):
        self.path1 = PATH1
        self.path2 = PATH2
        self.num = TRAIN_NUM
        self.savename = "svm{}.mat".format(SVMNAME)
        self.svm = None
        self.labels = None
        self.samples = None
        #裁剪的ROI参数
        self.cuty0 = HEIGHT*2//5
        self.cuty1 = HEIGHT*4//5
        self.cutx0 = WIDTH//3
        self.cutx1 = WIDTH*2//3
    #加载数据集并添加标签
    def data_load(self,path,num,act_label,samples,labels):
        for i in range(1,num):
            src_img=cv2.imread(path+str(i)+'.jpg')
            gray_img=cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)
            resize_img=cv2.resize(gray_img,(20,20))
            temp=np.resize(resize_img,(1,20*20))
            description=temp.astype(np.float32)
            samples.append(description)
            labels.append(act_label)
    #图像预处理
    def img_pre_process(self,image):
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resize_img = cv2.resize(gray_img, (20, 20))
        temp = np.resize(resize_img, (1, 20 * 20))
        description=temp.astype(np.float32)
        return description
    #裁剪目标区域
    def cut_img(self,image):
        cut_image=image[self.cuty0:self.cuty1,self.cutx0:self.cutx1]    #裁剪ROI
        cv2.imshow('1',cut_image)
        return cut_image
    #圈出轮廓矩形
    def get_rectangle(self,image):   
        #圈出最大轮廓
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,5,4)

        contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        t=()
        
        t = max(contours,key=cv2.contourArea)
        cv2.drawContours(image,t,-1,(0,255,0))
        cv2.imshow('1',binary)
        x, y, w, h = cv2.boundingRect(t)

        return x,y,w,h
    #对获取的帧的ROI进行分类
    def train(self):
        if os.path.exists(self.savename):
            try:
                self.svm = cv2.ml.SVM_load(self.savename)
                print("模型加载成功!")
            except Exception:
                print("导入失败!")
                exit()
        else:
            #创建一个SVM分类器
            self.svm=cv2.ml.SVM_create()
            #参数设置
            self.svm.setType(cv2.ml.SVM_C_SVC)
            self.svm.setKernel(cv2.ml.SVM_LINEAR)
            self.svm.setP(0.1)
            criterial = (cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
            self.svm.setTermCriteria(criterial)
            #数据集创建
            samples=[]
            labels=[]
            #加载数据集
            self.data_load(self.path1,self.num,0,samples,labels)
            self.data_load(self.path2,self.num,1,samples,labels)
            #将原始数据集（列表形式）转化为numpy数组形式，且需要是float32类型，因为SVM只接受这种类型
            samples_number = len(samples)
            samples = np.float32(samples)
            samples = np.resize(samples, (samples_number, 20*20))
            labels = np.int32(labels)
            labels = np.resize(labels, (samples_number, 1))
            print('training\n')
            #开始训练
            self.svm.trainAuto(samples,cv2.ml.ROW_SAMPLE, labels)
            self.svm.save(self.savename)
            print('ok')
            self.labels =labels
            self.samples = samples
    def start(self,frame):
        cut0=self.cut_img(frame)#初步裁剪出一个较小的区域
        try:
            x,y,w,h=self.get_rectangle(cut0)#找出框住字符的矩形
            cv2.rectangle(frame, (x+WIDTH//3, y+HEIGHT*2//5), (x + w+WIDTH//3, y + h+HEIGHT*2//5), (0, 0, 255), 3)#在最原始的图中绘制找到的矩形框
            cut1=cut0[y:y+h,x:x+w]#按找到的矩形裁剪cut0得到最终的字符
            test=self.img_pre_process(cut1)#将得到的图片按处理数据集的方法处理得到test
            result=self.svm.predict(test)#将test给SVM分类器预测
            if result[1][0][0]==0.0:
                print('WaitStop')
            elif result[1][0][0] == 1.0:
                print('Stop')

            cv2.imshow('video', frame)
        except Exception:
            pass
#############################################################################################  
#识别交通灯颜色    
class ColorRecognize():
    def __init__(self):
        #颜色区间
        self.color_dict = COLOR_DICT
        self.colors = COLORS
        self.color = None
        self.cropy0 = 200  
        self.cropy1 = 300
        self.cropx0 = 200
        self.cropx1 = 400
    #记录白色像素个数
    def get_white_pixel(self,img):
        return len(np.where(img == 255)[0])

    #通过计算哪种颜色的掩膜的白色像素数量最多确定颜色
    def get_color(self,img):
        try:
            img = img[self.cropy0:self.cropy1,self.cropx0:self.cropx1]  #缩小区域
            img_test = img.copy()   #在缩小后的区域找最亮区域
            img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(img_test)

            img_test2 = img.copy() #这里是为了显示出第一次缩小的图像，在图像中圈出最亮区域
            #在上次缩小的区域中截取出最亮点的区域作为目标区域
            img = img[maxLoc[1]-20:maxLoc[1]+20,maxLoc[0]-20:maxLoc[0]+20]

            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            #高斯模糊
            hsv = cv2.GaussianBlur(hsv,(5,5),0)
            maxcount = 0
            color = None
            for i in self.colors:
                mask = cv2.inRange(hsv,self.color_dict[i][0],self.color_dict[i][1])
                kernel = np.ones((3,3),np.uint8)
                #填补孔洞
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,iterations=5)
                res = cv2.bitwise_and(img,img,mask=mask)
                
                count = self.get_white_pixel(mask)
                if count > maxcount:
                    maxcount = count
                    if i == "highred" or i == "lowred":
                        i = "red"
                    color = i

            cv2.circle(img_test2, maxLoc, 20, (255, 0, 0), 2)
            #cv2.imshow('2',mask)
            cv2.imshow('1',img)
            cv2.imshow('3',img_test2)

            return color
        except Exception:
            pass
    def start(self,frame):
        print(self.get_color(frame))

#############################################################################################
#巡线
class Findroute():
    def __init__(self,direction = 'R'):
        #self.RouteRegionParams = [HEIGHT//2+130,HEIGHT//2+150,WIDTH//12+20,WIDTH-WIDTH//12-20]
        self.RouteRegionParams = [HEIGHT//2+130,HEIGHT//2+150,WIDTH//12+30,WIDTH-WIDTH//12-10]
        self.UpWaitstopRegionParams = [HEIGHT//2+60,HEIGHT-160,WIDTH//3-50,WIDTH*2//3+50]
        self.DownWaitstopRegionParams = [HEIGHT//2,HEIGHT//2+60,WIDTH//3-50,WIDTH*2//3+50]
        self.threshold = 130
        self.direction = direction
        pass
    def start(self,img):
        try:
        #转灰度图
            Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            [cropx1,cropx2,cropy1,cropy2] = self.RouteRegionParams   #裁剪的ROI参数
        
            Routeregion = Gray[cropx1:cropx2,cropy1:cropy2]    #ROI
            
            Routeregion = cv2.threshold(Routeregion, self.threshold, 255, cv2.THRESH_BINARY)[1]    #长方形条
            Gray = cv2.threshold(Gray, self.threshold, 255, cv2.THRESH_BINARY_INV)[1]    #原始灰度图

            #预处理ROI
            Routeregion = cv2.GaussianBlur(Routeregion,(5,5),0)
            kernel = np.ones((3,3),np.uint8)
            Routeregion = cv2.morphologyEx(Routeregion, cv2.MORPH_OPEN, kernel,iterations=1)
            Routeregion = cv2.morphologyEx(Routeregion, cv2.MORPH_CLOSE, kernel,iterations=1)
            
            #框出等停标志识别区域
            UpWaitstopRegion = Gray[self.UpWaitstopRegionParams[0]:self.UpWaitstopRegionParams[1],self.UpWaitstopRegionParams[2]:self.UpWaitstopRegionParams[3]]
            DownWaitstopRegion = Gray[self.DownWaitstopRegionParams[0]:self.DownWaitstopRegionParams[1],self.DownWaitstopRegionParams[2]:self.DownWaitstopRegionParams[3]]
            white = len(np.where(UpWaitstopRegion == 255)[0])
            black = len(np.where(DownWaitstopRegion == 0)[0])
            size1 = UpWaitstopRegion.size
            size2 = DownWaitstopRegion.size
            white_ratio = white/size1*100
            black_ratio = black/size2*100
            # if black_ratio>=90 and white_ratio>=20:
                # ser.write("w\r\n".encode('utf-8'))
            #print('bt,wt',black_ratio,white_ratio)
            # cv2.imshow('2',UpWaitstopRegion)
            # cv2.imshow('3',DownWaitstopRegion)

            #寻找ROI中黑线轮廓
            contours = cv2.findContours(Routeregion, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
            delta = np.array([[cropy1,cropx1]]) #变换坐标的参数(裁切后的图的坐标转化为原灰度图坐标)
            
            if self.direction == "R":
                contour = contours[1]   #与左右转弯有关，这是参考的轮廓
            elif self.direction == "L":
                contour = contours[-1] 
            contour += delta    #变换轮廓坐标
            x,y,w,h = cv2.boundingRect(contour)  #拟合为矩形以找出轮廓边界
                
            #根据轮廓边界确定线的中心点
            center1 = (x+w+10,y+h//2)
            center2 = (WIDTH//2,HEIGHT//2)

            #画出点
            cv2.drawContours(img, contour, -1, (255,0,0), 3)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.circle(img,center1,0,(0,0,255),10)
            cv2.circle(img,center2,0,(255,0,0),10)

            #图像中心点与目标点偏移距离
            direction = center1[0] - center2[0]
            if direction > 0:
                if direction >= 150:
                    direction = 150
                #ser.write('#{}#\r\n'.format(int(direction)).encode('utf-8'))
                print("右转")
            elif direction < 0:
                if direction <= -150:
                    direction = -150
                #ser.write('#{}#\r\n'.format(int(direction)).encode('utf-8'))
                print('左转')
            print('juli',direction)
            
            #截取出停止标志识别区域
            StopRegion = Gray[y+h//2-200:HEIGHT-250,x+w+10-50:x+w+10+50]

            white_pixels = len(np.where(StopRegion == 255)[0])
            pixels = StopRegion.size
            white_ratio = white_pixels/pixels*100
            if white_ratio>=50 and white_ratio<=65:
                #ser.write("s\r\n".encode('utf-8'))
                pass
            print('stopwt',white_ratio)
            
            cv2.rectangle(img,(cropy1,cropx1),(cropy2,cropx2),(0,255,0),3)
            cv2.imshow('1',img) 
            time.sleep(0.04)
        except Exception:
            pass
#############################################################################################

A = WaitandStop()
# B = ColorRecognize()
#C = Findroute()
A.train()
while True:
    ret, frame = capture.read()
    A.start(frame)
    #B.start(frame)
    #C.start(frame)
    if cv2.waitKey(10) == 27:
        break
cv2.destroyAllWindows() 


    




