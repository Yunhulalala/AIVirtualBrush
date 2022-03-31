# @File  : VirtualBrush.py
# @Date  :  2021/06/01

import cv2
import numpy as np
import os
import HandTrackingMoudle2 as htm

# 导入画板图片
footpath = "palette"
myPathList = os.listdir(footpath)
print(myPathList)
overLayList = []
for imagePath in myPathList:
    img = cv2.imread(f'{footpath}/{imagePath}')
    overLayList.append(img)
print(len(overLayList))
header = overLayList[0]
# 画笔颜色
color = (179, 230, 40)
# 新建画布
drawBroad = np.zeros((720, 1280, 3), dtype=np.uint8)
# 画笔厚度
penThickness = 10
# 橡皮擦厚度
eraserThickness = 40
cap = cv2.VideoCapture(0)
xp, yp = 0, 0
# 设置相机的分辨率为1280 × 720
cap.set(3, 1280)
cap.set(4, 720)

# 获取手部检测器
detector = htm.handDetector()
if not cap.isOpened():
    print("Can not open the Camera")
    exit()
while True:
    # 1. 获取图像
    ret, frame = cap.read()
    # 翻转图像
    frame = cv2.flip(frame, 1)
    # 2. 识别手势，获取手的landmark
    frame = detector.findHands(frame)
    landMarkList = detector.findPosition(frame)
    # 3.识别举起来的手指
    fingers = detector.findFingerUp()
    # 将画板放置于屏幕上方
    frame[0:180, 0:1280] = header
    count = 0
    if fingers:
        for i in range(0, 5):
            if fingers[i] == 1:
                count += 1
        # print(count, fingers)
        # 3 若举起来有两根手指，分别是食指和中指，进入选择模式
        if count == 2 and fingers[1] == 1 and fingers[2] == 1:
            # 食指指尖的序号在landMarkList为8，中值指尖的序号在landMarkList为12
            x1, y1 = landMarkList[8][0], landMarkList[8][1]
            x2, y2 = landMarkList[12][0], landMarkList[12][1]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            # 3.1 选择到了调色板区域
            if y1 < 180 and y1 > 0:
                # 绿色画笔
                if 0 < x1 and x1 < 256:
                    color = (179, 230, 40)
                    header = overLayList[0]
                # 蓝色画笔
                elif 257 < x1 and x1 < 512:
                    color = (200, 144, 46)
                    header = overLayList[1]
                # 红色画笔
                elif 513 < x1 and x1 < 768:
                    color = (87, 87, 255)
                    header = overLayList[2]
                # 黄色画笔
                elif 767 < x1 and x1 < 1024:
                    color = (0, 241, 244)
                    header = overLayList[3]
                # 橡皮擦
                elif 1025 < x1 and x1 < 1280:
                    color = (0, 0, 0)
                    header = overLayList[4]
            # 进入选择模式，则将xp yp置为0
            xp = 0
            yp = 0
        # 4 若举起来有1根手指，且为食指，进入画图模式
        elif count == 1 and fingers[1] == 1:
            x0, y0 = landMarkList[8][0], landMarkList[8][1]
            cv2.circle(frame, (x0, y0), 20, color, -1)
            if xp == 0 and yp == 0:
                xp, yp = x0, y0
            # 选择了橡皮擦模式
            if color == (0,0,0):
                cv2.line(drawBroad, (xp, yp), (x0, y0), color, eraserThickness)
            else:
                cv2.line(drawBroad, (xp, yp), (x0, y0), color, penThickness)
            xp = x0
            yp = y0
    # 5 将画板上笔迹添加到Camera的界面中
    # 灰度化
    imgGray = cv2.cvtColor(drawBroad,cv2.COLOR_BGR2GRAY)
    # 二值逆运算，画布笔迹部分为黑，背景部分为白
    ret, imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    # 按位与，frame的笔迹部分为黑，背景部分保留
    frame = cv2.bitwise_and(frame,imgInv)
    # 按位或，frame的黑色笔迹覆盖上画布的彩色
    frame = cv2.bitwise_or(frame,drawBroad)
    # cv2.imshow("DrawBroad", drawBroad)
    cv2.imshow("Camera", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
