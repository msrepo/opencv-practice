
from PIL import Image
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import numpy as np
import img_operators as imops
from img_operators import *

def show_images(images: list, grayscale = False,title=None) -> None:
    n: int = len(images)
    f = plt.figure()
    for i in range(n):
        # Debug, plot figure
        f.add_subplot(1, n, i + 1)
        if grayscale:
            plt.imshow(images[i],cmap='gray')
        else:
            plt.imshow(images[i])
        plt.axis('off')

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show(block=True)

    
    
def main():
    img = Image.open('../sample_videos/Lenna.png')
    r,g,b = img.split()
    img2 = Image.merge('RGB',(g,b,r))
##    show_images([r,g,b],grayscale=True,title="R G B Components")
    img3 = nd.shift(img, (0,0,1))
    y,i,q = imops.rgb_yiq(img)
##    show_images([y,i,q],grayscale=True)
##    show_images([create_rect((512,512))],grayscale=True)
##    show_images([create_circle((512,512),(256,256),100)],grayscale=True)

                  
        
    
if __name__ == '__main__':
    main()
    


