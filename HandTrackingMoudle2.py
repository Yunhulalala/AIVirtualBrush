# @File  : HandTrackingMoudle2.py
# @Author: Zeng Yixuan
# @Date  :  2021/06/02
import cv2
import mediapipe as mp


class handDetector():
    """
    手部监测器
    """

    def __init__(self, static_image_mode=False,
                 max_num_hands=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        初始化属性
        """
        # 静态图像模式，如果值为False，则输入图像为视频流。
        self.mode = static_image_mode
        # 最大检测手的数量
        self.max_num_hands = max_num_hands
        # 最小手部检测置信值
        self.min_detection_confidence = min_detection_confidence
        # 最小地标跟踪模型置信值
        self.min_tracking_confidence = min_tracking_confidence
        # 初始化手部检测器
        self.hands = mp.solutions.hands.Hands(self.mode,
                                         self.max_num_hands,
                                         self.min_tracking_confidence,
                                         self.min_tracking_confidence)
        self.results = None
        self.landmarkList = None
        # 五个手指指尖的序号
        self.fingersTip =[4,8,12,16,20]

    def findHands(self, img, draw=True):
        """
        :param img: 图像
        :return: 返回被标记21个地标的手的图像
        """
        #  将 BGR 图像转换为 RGB。
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        # 处理 RGB 图像并返回每只检测到的手的手部手部地标和手性
        # results.multi_hand_landmarks检测到的每只手的手部手部地标
        # results.handedness 检测到的每个手手的手性
        self.results = self.hands.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # MULTI_HAND_LANDMARKS：收集被检测手部的信息集合，
        # 其中每只手表示为 21 个手标志的列表，每个标志由 x、y 和 z 组成。
        if self.results.multi_hand_landmarks:
            # 遍历检测到的手
            for hand in self.results.multi_hand_landmarks:
                if draw:
                    # 在图像上绘制和连接地标
                    mp.solutions.drawing_utils.draw_landmarks(img, hand, mp.solutions.hands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img,handNo=0,draw=True):

        self.landmarkList = []
        if self.results.multi_hand_landmarks:
            # 获取指定手的信息
            myhand = self.results.multi_hand_landmarks[handNo]
            for lm in myhand.landmark:
                h, w, c = img.shape
                # lm.x lm.y 的值是经过归一化的
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.landmarkList.append([cx,cy])
                # 用灰色圆圈标识指定手的位置
                if draw:
                    cv2.circle(img,(cx,cy),7,(196,196,196),-1)
        return self.landmarkList

    def findFingerUp(self):
        """
        检测举起来的手指（仅对于右手而言,且手掌方向朝外）
        :return: 返回一个列表，表示五根手指的状态，0表示没有举起来，1表示举起来
        """
        fingers =[]
        if self.landmarkList:
            # 大拇指的检测，若4点的x坐标小于3点的x坐标，则被认为大拇指举起来了
            if self.landmarkList[self.fingersTip[0]][0] < self.landmarkList[self.fingersTip[0]-1][0]:
                fingers.append(1)
            else:
                fingers.append(0)
            # 对于剩余四个拇指的检测，若8、12、16、20的y坐标小于对应6、10、14、18的y坐标，则被认为举起来了
            for i in range(1,5):
                if self.landmarkList[self.fingersTip[i]][1] < self.landmarkList[self.fingersTip[i]-2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    # 实例化hanDetector对象
    detector = handDetector()
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.flip(image,1)
        if not success:
            print("Failure!!!")
            continue
        image = detector.findHands(image)
        lm = detector.findPosition(image)
        finger = detector.findFingerUp()
        if finger:
            print(finger)
        # if lm:
        #     print(lm)
        cv2.imshow("Camera", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
