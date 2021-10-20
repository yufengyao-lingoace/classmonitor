import numpy as np
import os
import cv2
import onnxruntime
class FacialExpressionDetector:
    def __init__(self):
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 4 #设置线程数
        opts.inter_op_num_threads = 4 #设置线程数
        opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        self.onnx_session=onnxruntime.InferenceSession(os.path.join(os.path.dirname(__file__),'weights', 'facialexpression.onnx'),opts) ##ONNX模式检测
        self.class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    def detect(self,gray):    
        try:
            gray=cv2.resize(gray, (44, 44), interpolation=cv2.INTER_LINEAR)
            img = gray[:, :, np.newaxis]
            img=img/255
            img = np.concatenate((img, img, img), axis=2)
            img=img.transpose(2,0,1)
            inputs=[]
            # for i in range(10):
            inputs.append(img)
            inputs=np.array(inputs,dtype=np.float32)
            # ncrops=10
            inputs={self.onnx_session.get_inputs()[0].name:inputs}
            outs=self.onnx_session.run(None,inputs)
            outputs=np.array(outs[0])
            x=outputs[0]
            # ###求softmax
            x = x - x.max(axis=None, keepdims=True)
            y = np.exp(x)
            score= y / y.sum(axis=None, keepdims=True)
            predicted=np.argmax(score)
            emojis=self.class_names[predicted]
            score=score.tolist()
            return [emojis,score]
        except Exception as ex:
            print('emotion ana error:'+str(ex))
            return []