import argparse
import cv2
from inference import Network
import numpy as np

MODEL_PATH = "./model_zoo/"

FACE_MODEL_FILE = MODEL_PATH+"intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml"
FACE_MODEL_FILE2 = MODEL_PATH+"intel/face-detection-retail-0004/FP16/face-detection-retail-0004.xml"
CPU_EXTENSION = "/opt/intel/openvino_2019.3.376/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"


def arg_parse():
    parser = argparse.ArgumentParser("Detect Face")

    parser.add_argument('-i',help = "location of the input file",required=True)
    parser.add_argument('-m',help = "location of the model xml file",default=FACE_MODEL_FILE)
    parser.add_argument('-d',help = "device type",default='CPU')

    return parser.parse_args()

def preprocessing(image, c, h, w):
    image = np.copy(image)
    image = image.transpose((2,0,1))
    image = image.reshape((1,c,h,w))
    return image
    
    
def get_result(image,model):
    model.async_inference(image)
    if model.wait() == 0:
        result = model.extract_output()
    result = postprocessing(result)
    return result

def postprocessing(result):
    result = result[0,0] # superfluos first two channels 1x1xNx7
    return result[np.where(result[:,2] > 0.95)] # 95% confidence
    
def bounding_boxes(image,result):
    if result.size == 0:
        return image # when there are no faces in the image
 
    h,w,c = image.shape 
    #print("image shape ",image.shape)
    for i in range(len(result)):
        result_i = result[i]
        topxy = (int(result_i[3]*w),int(result_i[4]*h))
        bottomxy = (int(result_i[5]*w),int(result_i[6]*h))
        image =  cv2.rectangle(image, topxy, bottomxy, (255,0,0),2)
    return image

def infer_on_video(args):
    plugin = Network()
    plugin.load_model(args.m,args.d,CPU_EXTENSION)
    b,c,h,w = plugin.get_input_shape()
    # print("Required shape:",b,c,h,w)

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
        
        result = get_result(r_frame,plugin)
        output = bounding_boxes(frame,result)
                             
        
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
