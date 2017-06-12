import pandas as pd
from pandas import DataFrame
from pandas import Series
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def makeMat(str_arr, size=(48,48)):
    vec = np.array(str_arr.split()).astype(float)
    return vec.reshape(48,48)

print('Loading images...')
fer_images = pd.read_csv('/Users/lab/Documents/Datasets/fer2013/fer2013.csv')
print('Done.')


fer_images['mat'] = fer_images['pixels'].map(lambda x: makeMat(x))
for i in fer_images['mat'].head().values:
    print(i)
