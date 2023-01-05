import cv2 as cv

capture = cv.VideoCapture(0)
capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
path = 'C:/Users/4869/Desktop/Car_project/L_CUT'
i = 1
while True:
    ret, frame = capture.read()
    
    cv.imshow('video', frame)
    
    if cv.waitKey(10) == 13:
        if i == 51:
            break
        cv.imwrite(path+'img3-'+str(i)+'.jpg',frame)
        print("ok")
        i += 1
cv.destroyAllWindows()  
