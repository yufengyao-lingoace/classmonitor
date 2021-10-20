import os
import sys
sys.path.append(os.path.dirname(__file__))
from HeadPoseCal import HeadPoseCal
from FacialExpression import FacialExpressionDetector
import numpy as np
import dlib
import cv2
import eye_utils
from retinaface_onnx import Retinaface_onnx
class OneImageAna:
    def __init__(self):
        self.retinaface_detector=Retinaface_onnx() #retinaface人脸检测
        self.facialdector = FacialExpressionDetector()  # 表情检测
        self.predictor = dlib.shape_predictor(os.path.join(os.path.dirname(__file__), 'weights/shape_predictor_68_face_landmarks.dat'))  # 68个关键点检测

    def getResult(self, frame, border=0): #retinaface
            result = []
            face_num=0
        # try:
            gray_ori = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bboxes=self.retinaface_detector.detect(frame)
            for face in bboxes:
                face=dlib.rectangle(int(face[0]),int(face[1]),int(face[2]),int(face[3]))
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
                yaw, pitch, roll,nose_end_point2D = HeadPoseCal.getHeadPoseAngle(frame.shape, keypoints) #计算角度
                eyepose = HeadPoseCal.getEyePose(keypoints)
                mouse_ratio,mouse_distance = HeadPoseCal.getMousePose(keypoints)
                # obj["yaw"] = yaw
                # obj["pitch"] = pitch
                # obj["roll"] = roll
                eyeball_x,eyeball_y=eye_utils.get_eyeball(frame,obj["keypoints"])
                obj["eye"] = [eyepose]
                obj["eye"].append(eyeball_x)
                obj["eye"].append(eyeball_y)
                obj["nose_end"]=nose_end_point2D[0][0]
                obj["mouse"] = [mouse_ratio,mouse_distance]
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


    