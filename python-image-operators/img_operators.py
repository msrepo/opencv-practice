
import numpy as np
import scipy.ndimage as nd

def create_rect(imgsize, width = 256, height = 256):
    h,w = imgsize
    img = np.zeros(imgsize)
    topleft = (h - height) // 2, (w - width) // 2
    img[topleft[0]:topleft[0]+height,topleft[1]: topleft[1] + width] = 1
    return img

def create_circle(imgsize, center, radius):
    x,y = np.indices(imgsize)
    x,y = x - center[0], y - center[1]
    mask = x**2 + y**2
    mask = (mask <= radius**2).astype(int)
    return mask
    

def rgb_grayscale(img):
    img = np.array(img)
    return 0.5 * img[:,:,0] + 0.75 * img[:,:,1] + 0.25 * img[:,:,2]

def rgb_yiq(img):
    img = np.array(img)
    R, G, B = tuple(map(lambda x: img[:,:,x], [0,1,2]))
    Y = 0.299*R + 0.587*G + 0.114*B
    I = 0.596*R - 0.275*G - 0.321*B
    Q = 0.212*R - 0.523*G + 0.311*B
    print(arg_max_energy(np.array([Y,I,Q])))
    return (Y,I,Q)

def avg_thresholded(img, threshold = 0):
    img = np.array(img)
    den = (img > threshold).sum()
    num = ((img > threshold) * img).sum()
    return num / den

def arg_max_energy(img):
    img = np.array(img)
    h,w,c = img.shape
    energy = map(lambda x: (img[:,:,x] ** 2).sum()/(h*w) , [0,1,2])
    return np.argmax(energy)
    
    
    
