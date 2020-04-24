import cv2
from inference import Network
import numpy as np


def load_head_pose_model(arg_m,arg_d,args_o):
    plugin = Network()
    plugin.load_model(arg_m,arg_d,args_o)
    b,c,h,w = plugin.get_input_shape()
    print("Required Shape:",b,c,h,w)
    return plugin

    
def preprocessing(image, c, h, w):
    image = np.copy(image)
    image = image.transpose((2,0,1))
    image = image.reshape((1,c,h,w))
    return image

def postprocessing(model_type, result):
    if model_type == "POSE":
        return result
    elif model_type == "FACE":
        result = result[0,0] # superfluos first two channels 1x1xNx7
        return result[np.where(result[:,2] > 0.95)] # 95% confidence

	
def bounding_boxes(image,result):
    if result.size == 0:
        return image,((0,0),(0,0)) # when there are no faces in the image
 
    h,w,c = image.shape 
    #print("image shape ",image.shape)
    for i in range(len(result)):
        result_i = result[i]
        topxy = (int(result_i[3]*w),int(result_i[4]*h))
        bottomxy = (int(result_i[5]*w),int(result_i[6]*h))
        lb = (bottomxy[0]-topxy[0], bottomxy[1] - topxy[1])
        image =  cv2.rectangle(image, topxy, bottomxy, (255,0,0),2)
        #print("Face bounding box:",topxy,lb)
    return image,(topxy,bottomxy)

def get_result(model_type, image,model):
    model.async_inference(image)
    if model.wait() == 0:
        result = model.extract_output()
    result = postprocessing(model_type, result)
    return result

def drawAxis(frame,rot):
    axis = np.float32([[1,0,0], [0,1,0], [0,0,0],[100,0,0], [0,100,0], [0,0,0.5]]).reshape(-1,3)
    mtx, dist = np.eye(3), np.float64([])
    


    rvec, tvec = np.zeros(3), np.array([50,50,0.0])
    imgpts, jac = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
    cv2.line(frame, tuple(imgpts[3].ravel()), tuple(imgpts[0].ravel()), (0,0,255), 2)
    cv2.line(frame, tuple(imgpts[4].ravel()), tuple(imgpts[1].ravel()), (0,255,0), 2)
    cv2.line(frame, tuple(imgpts[5].ravel()), tuple(imgpts[2].ravel()), (255,0,0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'X', tuple(imgpts[3].ravel()), font, 0.5, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(frame, 'Y', tuple(imgpts[4].ravel()), font, 0.5, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(frame, 'Z', tuple(imgpts[5].ravel()), font, 0.5, (255,0,0), 2, cv2.LINE_AA) 
