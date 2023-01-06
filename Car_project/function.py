import cv2
import numpy as np
import os
import time

WIDTH = 640
HEIGHT = 480

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

COLOR_DICT = {
                'highred':[np.array([156,43,46]),np.array([180,255,255])],
                'lowred':[np.array([0,43,46]),np.array([10,255,255])],
                'green':[np.array([35,43,46]),np.array([77,255,255])],
                'blue':[np.array([100,43,46]),np.array([124,255,255])],
                'yellow':[np.array([26,43,46]),np.array([34,255,255])],
                'orange':[np.array([11,43,46]),np.array([25,255,255])]}
COLORS = ['highred','lowred','green','blue','yellow','orange']
 
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
            cv2.rectangle(img,(cropy1,cropx1),(cropy2,cropx2),(0,255,0),3)
            cv2.imshow('1',img) 
            time.sleep(0.04)
        except Exception:
            pass
#############################################################################################

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


    




