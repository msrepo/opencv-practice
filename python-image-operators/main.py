
from PIL import Image
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import numpy as np
import img_operators as imops
from img_operators import *
from gabor import *

def show_images(images: list, grayscale = False,title=None,nrows = 1) -> None:
    n: int = len(images)
    f = plt.figure()
    for i in range(n):
        # Debug, plot figure
        f.add_subplot(nrows, n // nrows , i + 1)
        if grayscale:
            plt.imshow(images[i],cmap='gray')
        else:
            plt.imshow(images[i])
        plt.axis('off')

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show(block=True)


def show_gabor_filterbanks():
    theta = np.arange(0,np.pi, np.pi/4)
    sigma = np.arange(1,5)
    params = [(t,s) for t in theta for s in sigma]
    filterbanks = []
    for t,s in params:
        gabor = gen_gabor((128,128),theta = t,sigma = s,gamma = 2)
        filterbanks.append(gabor)
    show_images(filterbanks,True,nrows=4)
        
    
def main():
    img = Image.open('../sample_videos/Lenna.png')
    r,g,b = img.split()
    img2 = Image.merge('RGB',(g,b,r))
    img3 = nd.shift(img, (0,0,1))

    theta = np.pi / 2
    omega = [np.cos(theta),np.sin(theta)]
    phase = np.pi/2
    #show_images([gen_sinusoid((256,256),1,omega,phase)],True)
    #show_images([gen_gabor((128,128),gamma = 2, Lambda = 0.1)])
    #show_gabor_filterbanks()
    img = Image.open('./tools_dataset/01_01.bmp')
    # show_images([img],True)

    

                  
        
    
if __name__ == '__main__':
    main()
    


