
import sys
import os

from numpy.lib import utils
sys.path.append(os.path.dirname(__file__))
from HeadPoseCal import HeadPoseCal
from FacialExpression import FacialExpressionDetector
import numpy as np
import dlib
import cv2
# from collections import OrderedDict
import eye_utils

class OneImageAna:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()  # dlib人脸检测
        self.facialdector = FacialExpressionDetector()  # 表情检测
        self.predictor = dlib.shape_predictor(
            os.path.join(os.path.dirname(__file__), 'weights/shape_predictor_68_face_landmarks.dat'))  # 68个关键点检测

    def getResult(self, frame, border=0):
            result = []
            face_num=0
        # try:
            gray_ori = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # frame=cv2.resize(frame, (240, 240), interpolation=cv2.INTER_LINEAR)
            # gray_ori = frame
            bboxes, score, idx = self.detector.run(gray_ori, 0, 0)  # 不加边图像识别
            # if len(bboxes)==0:
            # bboxes=self.detector(gray_ori,1)
            # frame = cv2.copyMakeBorder(frame, border, border, border, border, cv2.BORDER_CONSTANT, value=[255, 255, 255])  # 图像边缘扩展 0.00023s
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            for face in bboxes:
                if face.height() < 20 or face.width() < 20:
                    continue
                face_num+=1
                face = dlib.rectangle(face.left()+border, face.top()+border, face.right()+border, face.bottom()+border)  # 加边
                obj = {}
                # obj["face"] = face
                obj["face"] = [(face.left(),face.top()),(face.right(),face.bottom())]
                faceimg = gray_ori[face.top():face.bottom(),face.left():face.right()]
                if faceimg.shape[0] < 5 or faceimg.shape[1] < 5:
                    continue
                emoji = self.facialdector.detect(faceimg) #表情识别
                obj["emo"] = list(emoji)
                keypoints = self.predictor(gray_ori, face)  #68个点
                # obj["keypoints"] = keypoints
                obj["keypoints"]=[[p.x, p.y] for p in keypoints.parts()]
                # yaw, pitch, roll = HeadPoseCal.getHeadPoseAngle(frame.shape, keypoints)
                eyepose = HeadPoseCal.getEyePose(keypoints)
                mouse_ratio,mouse_distance = HeadPoseCal.getMousePose(keypoints)
                # obj["yaw"] = yaw
                # obj["pitch"] = pitch
                # obj["roll"] = roll
                eyeball_x,eyeball_y=eye_utils.get_eyeball(frame,obj["keypoints"])
                obj["eye"] = [eyepose]
                obj["eye"].append(eyeball_x)
                obj["eye"].append(eyeball_y)

                obj["mouse"] = [mouse_ratio,mouse_distance]
                # obj['size'] = face.height()*face.width()
                result.append(obj)
            if len(result) > 0:  # 多个面部筛选
                max_size,true_index=0,0
                for index,item in enumerate(result):
                    # f_size = item["face"].height()*item["face"].width()
                    f_size=abs((item["face"][0][0]-item["face"][1][0])*(item["face"][0][1]-item["face"][1][1]))
                    if f_size > max_size :
                        true_index=index
                        max_size = f_size
                result=[result[true_index]]
        # except Exception as ex:
        #     print(ex)
        # finally:
            return result,face_num



    