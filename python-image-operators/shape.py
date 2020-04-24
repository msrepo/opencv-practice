import numpy as np
import scipy.ndimage as nd

def perimeter_points(img,pick = 0):
    img = img[:,:,0]
    eroded = nd.binary_erosion(img)
    vidx,hidx = eroded.nonzero()
