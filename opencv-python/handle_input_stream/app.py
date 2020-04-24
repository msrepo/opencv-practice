import cv2
import argparse
import numpy as np

def capture_stream(args):
    image_flag = False
    if args.i.endswith('jpg') or args.i.endswith('bmp'):
        image_flag = True
    elif args.i == 'CAM':
        args.i = 0
        
    cap = cv2.VideoCapture(0)
    cap.open(args.i)

    if not image_flag:
        out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'XVID'),30,(200,200))
    else:
        out = None
        
    while cap.isOpened():
        flag,frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        frame = cv2.resize(frame,(200,200))
        frame = cv2.Canny(frame,100,200)
        frame = np.dstack((frame,frame,frame))
        if image_flag:
            cv2.imwrite('output-image.jpg',frame)
        else:
            out.write(frame)
        if key_pressed == 27:
            break
    if not image_flag:
        out.release()
        
    cap.release()
    cv2.destroyAllWindows()
    
def arg_parse():
    parser = argparse.ArgumentParser("Handle an input stream")
    i_desc = "the location of the input file"
    parser.add_argument("-i",help = i_desc)
    args = parser.parse_args()
    return args
                
def main():
    args = arg_parse()
    capture_stream(args)

if __name__ == "__main__":
    try:
        __file__
    except:
        sys.argv =[sys.argv[0], '-i' , 'bluecar.jpg']
    main()
