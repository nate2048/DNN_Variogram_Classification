import os
import numpy as np
import pandas as pd
from PIL import Image

def ImgData(filename):

    imgs = np.empty((len(os.listdir(filename)), 256**2))

    for i, name in enumerate(os.listdir(filename)):
        path = os.path.join(filename, name)
        image = Image.open(path).convert('L')
        image = np.array(image, dtype=np.float32) / 255.0

        imgs[i] = image.reshape(256**2)

    return imgs

def DiffusionData(filename):

    diffusion = pd.read_csv(filename, header = None)
    diffusion = diffusion.to_numpy()
    
    return diffusion
