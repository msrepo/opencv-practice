import argparse
import cv2
from inference import Network
import numpy as np
from handle_models import *

MODEL_PATH = "../../model_zoo/"

FACE_MODEL_FILE = MODEL_PATH+"intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml"
FACE_MODEL_FILE2 = MODEL_PATH+"intel/face-detection-retail-0004/FP16/face-detection-retail-0004.xml"
CPU_EXTENSION = "/opt/intel/openvino_2019.3.376/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
HEAD_POSE_MODEL = MODEL_PATH+"intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml"
DEVICE_CPU = "CPU"

def arg_parse():
    parser = argparse.ArgumentParser("Detect Face")

    parser.add_argument('-i',help = "location of the input file",required=True)
    parser.add_argument('-m',help = "location of the model xml file",default=FACE_MODEL_FILE)
    parser.add_argument('-d',help = "device type",default= DEVICE_CPU)

    return parser.parse_args()


def infer_on_video(args):
    face_model = Network()
    face_model.load_model(args.m,args.d,CPU_EXTENSION)
    b,c,h,w = face_model.get_input_shape()
    # print("Required shape:",b,c,h,w)
    head_pose_model = load_head_pose_model(HEAD_POSE_MODEL,DEVICE_CPU,CPU_EXTENSION)
    

    if args.i == 'CAM':
        args.i = 0
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    while cap.isOpened():
        flag,frame = cap.read()
        if not flag:
            break
        frame = cv2.resize(frame,(w,h))
        r_frame = preprocessing(frame,c,h,w)
        f_result = get_result("FACE",r_frame,face_model)
        output,(topxy,bottomxy) = bounding_boxes(frame,f_result)
        
        if bottomxy[0] - topxy[0] != 0:
            face_section = frame[topxy[1]:bottomxy[1],topxy[0]:bottomxy[0]].copy()
            
            rface_section = cv2.resize(face_section,(60,60))
            rface_section = preprocessing(rface_section,3,60,60)
            p_result = get_result("POSE",rface_section,head_pose_model)
            print(p_result.shape , p_result)
            
            cv2.imshow('face',face_section)
                             
            drawAxis(output,p_result)
        cv2.imshow("input", output)
        if cv2.waitKey(1000 // int(cap.get(5))) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    
    
def main():
    args = arg_parse()
    infer_on_video(args)


if __name__ == '__main__':
    main()
