# 输入图像储存和面部68个关键点，计算头部姿势角度
import cv2
import numpy as np
import time
import math


class HeadPoseCal():
    @staticmethod
    def get_image_points_from_landmark_shape(landmark_shape):
        image_points = np.array([
            (landmark_shape.part(30).x, landmark_shape.part(30).y),
            (landmark_shape.part(8).x, landmark_shape.part(8).y),
            (landmark_shape.part(36).x, landmark_shape.part(36).y),
            (landmark_shape.part(45).x, landmark_shape.part(45).y),
            (landmark_shape.part(48).x, landmark_shape.part(48).y),
            (landmark_shape.part(54).x, landmark_shape.part(54).y)
        ], dtype="double")
        return image_points

    @staticmethod
    def get_pose_estimation(img_size, image_points):
        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner

        ])
        focal_length = img_size[1]
        center = (img_size[1] / 2, img_size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points,
         image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        
        return success, rotation_vector, translation_vector, camera_matrix ,dist_coeffs 

    @staticmethod
    def get_euler_angle(rotation_vector):
        # calculate rotation angles
        theta = cv2.norm(rotation_vector, cv2.NORM_L2)
        # transformed to quaterniond
        w = math.cos(theta / 2)
        x = math.sin(theta / 2) * rotation_vector[0][0] / theta
        y = math.sin(theta / 2) * rotation_vector[1][0] / theta
        z = math.sin(theta / 2) * rotation_vector[2][0] / theta
        ysqr = y * y
        # pitch (x-axis rotation)
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + ysqr)
        # print('t0:{}, t1:{}'.format(t0, t1))
        pitch = math.atan2(t0, t1)
        # yaw (y-axis rotation)
        t2 = 2.0 * (w * y - z * x)
        if t2 > 1.0:
            t2 = 1.0
        if t2 < -1.0:
            t2 = -1.0
        yaw = math.asin(t2)
        # roll (z-axis rotation)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (ysqr + z * z)
        roll = math.atan2(t3, t4)
        # print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))
        Y = int((pitch / math.pi) * 180)
        X = int((yaw / math.pi) * 180)
        Z = int((roll / math.pi) * 180)
        return 0, Y, X, Z

    @staticmethod
    def getHeadPoseAngle(picsize, facepoints):  # 头部角度计算
        facepoints = HeadPoseCal.get_image_points_from_landmark_shape(facepoints)
        ret, rotation_vector, translation_vector, camera_matrix ,dist_coeffs= HeadPoseCal.get_pose_estimation(picsize, facepoints)
        ret, pitch, yaw, roll = HeadPoseCal.get_euler_angle(rotation_vector)
        pitch=180-(360+pitch)%360
        if(abs(roll)>90):
            roll = (180-abs(roll)) if roll > 0 else (abs(roll)-180)

        #计算鼻头朝向
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 340.0)]),rotation_vector,translation_vector,camera_matrix,dist_coeffs) #340表示图像尺寸
        return (yaw, pitch, roll,nose_end_point2D)
    @staticmethod
    def getEyePose(facepoints):#眼睛闭合度计算
        x36=facepoints.part(36).x
        y36=facepoints.part(36).y
        x37=facepoints.part(37).x
        y37=facepoints.part(37).y
        x38=facepoints.part(38).x
        y38=facepoints.part(38).y
        x39=facepoints.part(39).x
        y39=facepoints.part(39).y
        x40=facepoints.part(40).x
        y40=facepoints.part(40).y
        x41=facepoints.part(41).x
        y41=facepoints.part(41).y

        x42=facepoints.part(42).x
        y42=facepoints.part(42).y
        x43=facepoints.part(43).x
        y43=facepoints.part(43).y
        x44=facepoints.part(44).x
        y44=facepoints.part(44).y
        x45=facepoints.part(45).x
        y45=facepoints.part(45).y
        x46=facepoints.part(46).x
        y46=facepoints.part(46).y
        x47=facepoints.part(47).x
        y47=facepoints.part(47).y

        t1=math.sqrt((x37-x41)**2+(y37-y41)**2)
        t2=math.sqrt((x38-x40)**2+(y38-y40)**2)
        t3=math.sqrt((x36-x39)**2+(y36-y39)**2)
        r1=(t1+t2)/t3/2
        t4=math.sqrt((x43-x47)**2+(y43-y47)**2)
        t5=math.sqrt((x44-x46)**2+(y44-y46)**2)
        t6=math.sqrt((x42-x45)**2+(y42-y45)**2)
        r2=(t4+t5)/t6/2
        return round((r1+r2)/2,2)

    @staticmethod
    def getMousePose(facepoints):#嘴巴闭合度计算
        #内嘴角闭合度
        x61=facepoints.part(61).x
        y61=facepoints.part(61).y
        x67=facepoints.part(67).x
        y67=facepoints.part(67).y
        x63=facepoints.part(63).x
        y63=facepoints.part(63).y
        x65=facepoints.part(65).x
        y65=facepoints.part(65).y
        x60=facepoints.part(60).x
        y60=facepoints.part(60).y
        x64=facepoints.part(64).x
        y64=facepoints.part(64).y

        x51=facepoints.part(51).x
        y51=facepoints.part(51).y
        x57=facepoints.part(57).x
        y57=facepoints.part(57).y
        t1=math.sqrt((x61-x67)**2+(y61-y67)**2)
        t2=math.sqrt((x63-x65)**2+(y63-y65)**2)
        t3=math.sqrt((x60-x64)**2+(y60-y64)**2)
        distance=math.sqrt((x57-x51)**2+(y57-y51)**2)
        return round((t1+t2)/t3/2,2),int(distance)

