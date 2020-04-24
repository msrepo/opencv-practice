
import numpy as np

def gen_gabor(imgsize, sigma = 5.0, theta = np.pi / 4, Lambda = np.pi, psi = 0.0, gamma = 1.0):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 5  # Number of standard deviation 
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))
    
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gauss = np.exp(-0.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2))
    sinusoid = np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gauss * sinusoid


def gen_sinusoid(imgsize, A, omega, rho):
    ''' Generate sinusoid grating
    imgsize: size of the generated image (width, height)
    '''
    radius = (int(imgsize[0]/2.0),int(imgsize[1]/2.0))
    [x,y] = np.meshgrid(range(-radius[0],radius[0]+1),
                        range(-radius[1],radius[1]+1))
    stimuli = A * np.cos(omega[0] * x + omega[1] * y + rho)
    return stimuli



    
                              
