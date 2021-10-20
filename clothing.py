import numpy as np
import cv2
import dlib
import csv
from scipy.stats import mode

def skinMask_crcb(roi):
	YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB) #转换至YCrCb空间
	(y,cr,cb) = cv2.split(YCrCb) #拆分出Y,Cr,Cb值
	cr1 = cv2.GaussianBlur(cr, (5,5), 0)
	_, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #Ostu处理
	res = cv2.bitwise_and(roi,roi, mask = skin)
	return res


def detectForeheadPos(chin,nose):
    #计算截取视频区域的上边界
    xm,ym = chin
    xh,yh = nose
    xf = xh - (xm - xh)
    yf = yh - 3/4 * (ym - yh)
    xf = np.int0(xf)
    yf = np.int0(yf)
    return (xf,yf)

def detectKeyPoints(landmarks):
    xh = 0
    xm = 0
    ym = 0
    yh = 0
    fy = 0
    fx = 0
    lx = 0
    ly = 0
    rx = 0
    ry = 0
    
    # landmarks = predictor(inimg, face)
    # landmarks = np.matrix([[p.x, p.y] for p in landmarks.parts()])

    for idx, point in enumerate(landmarks):
        # 68点的坐标
        pos = (point[0], point[1])
        # 利用cv2.circle画出下巴边界点
        if idx == 8: 
            # cv2.circle(outimg, pos, 2, color=(255, 0, 0))
            chinpos = pos
            xm,ym = chinpos#下巴最低点坐标
        elif idx == 27:
            # cv2.circle(outimg, pos, 2, color=(255, 0, 0))
            nosepos = pos
            xh,yh = nosepos#鼻梁最高点坐标
        elif idx == 0:
            lx,ly = pos #面部左侧边界点
        elif idx == 16:
            rx,ry = pos #面部右侧边界点


    lx = lx + 1/4 * (lx - rx)   #图像左侧边界x
    rx = rx + 1/4 * (rx - lx)   #图像右侧边界x

    rx = np.int0(rx)
    ry = np.int0(ry)
    
    lx = np.int0(lx)
    ly = np.int0(ly)
    
    lpos = (lx,ly)
    rpos = (rx,ry)
    # cv2.circle(outimg, lpos, 2, color=(255, 0, 0))
    # cv2.circle(outimg, rpos, 2, color=(255, 0, 0))
    

    foreheadpos = detectForeheadPos(chinpos,nosepos)
    
    fx,fy = foreheadpos    #图像上边界
    
    # cv2.circle(outimg, foreheadpos, 2, color=(255, 0, 0))
    return fx,fy,lx,ly,rx,ry,xm,ym,xh,yh
        
# 教师位置看这里
t_width = 360
t_height = 360
t_X_pos = 0
t_Y_pos = 0    


def clothingExposureLevel(frame,landmarks,threshold, up_threshold = 2):
    ratio=-1
    height = t_height    #获取图像到下边界hight
    frame = cv2.resize(frame, (479, 620))
    skin = skinMask_crcb(frame)
    kernel = np.ones((3,3), np.uint8)
    erosion = cv2.erode(skin, kernel)
    dilation = cv2.dilate(erosion, kernel)
    dilation = cv2.dilate(dilation, kernel)
    dilation = cv2.dilate(dilation, kernel)
    dilation = cv2.dilate(dilation, kernel)

    fx,fy,lx,ly,rx,ry,xm,ym,xh,yh = detectKeyPoints(landmarks)
    
    top = int(fy)       #图像上边界纵坐标
    left = int(lx)      #左边界横坐标
    right = int(rx)     #右边界横坐标
    height = int(height)    #下边界纵坐标

    binaryimg = cv2.Canny(dilation, 50, 200) #二值化，canny检测
    binary,contours = cv2.findContours(binaryimg, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) #寻找轮廓
    # ret = np.ones(dilation.shape, np.uint8) #创建黑色幕布
    # cv2.drawContours(ret, binary,-1,(255,255,255),1) #绘制白色轮廓
    
    #寻找皮肤与领口衔接处标记点
    for cnt in binary:
        #绘制countour的直边外接矩形
        x,y,w,h = cv2.boundingRect(cnt)
        xl = x + 0.5 * w
        yl = y + h
        # 限制外接矩形大小，突出显示框出面部和脖子的部分。
        if w > 1/3 * (right - left) and h > (ym - yh) :
            # cv2.rectangle(ret,(x,y),(x+w,y+h),(0,255,0),2)#绘制面颈部皮肤的外接矩形
            xl = np.int0(xl)
            yl = np.int0(yl)
            # cv2.circle(ret, (xl, yl), 4, color=(255, 0, 0))#标记外接矩形底边中点
            tmp = (yl-ym+top)/(ym-yh)
            if tmp > threshold and tmp < up_threshold:
                ratio=tmp
    return ratio



