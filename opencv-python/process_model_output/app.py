import argparse
import cv2
from sys import platform
from inference import Network

INPUT_STREAM = 'pets.mp4'
def arg_parse():
    parser = argparse.ArgumentParser("Run inference on input video")
    i_desc = "location of input file"
    m_desc = "location of the model XML file"
    d_desc = "CPU"
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    
    parser.add_argument("-i",help = i_desc,default=INPUT_STREAM)
    parser.add_argument("-m",help = m_desc,required=True)
    parser.add_argument("-d",help = d_desc,default='CPU')
    args = parser.parse_args()
    return args

def incidence(result,counter, incidence_flag):
    if result[0][1] == 1 and not incidence_flag:
        incidence_flag = True
        timestamp = counter / 30
        print('Incidence occured at {:.2f} seconds\n'.format(timestamp))
    elif result[0][1] != 1:
        incidence_flag = False
    return incidence_flag
    
def infer_on_video(args):
    plugin = Network()
    plugin.load_model(args.m,args.d,None)
    net_input_shape = plugin.get_input_shape()

    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    counter = 0
    incidence_flag = False
    while cap.isOpened():
        flag,frame = cap.read()
        if not flag:
            break
        counter += 1
        r_frame = cv2.resize(frame, (net_input_shape[3],net_input_shape[2]))
        r_frame = r_frame.transpose((2,0,1))
        r_frame = r_frame.reshape(1,*r_frame.shape)

        plugin.async_inference(r_frame)
        if plugin.wait() == 0:
            result = plugin.extract_output()
            incidence_flag = incidence(result,counter,incidence_flag)        
    cap.release()
    cv2.destroyAllWindows()
        
def main():

    args = arg_parse()
    print(args.i,args.m)
    infer_on_video(args)
    
if __name__ == '__main__':
    main()
