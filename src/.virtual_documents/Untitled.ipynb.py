from PIL import Image, ImageFilter


im = Image.open("data/cell/3T3/high/DSC_0582.JPG")


im.show()


print(im.format, im.size, im.mode)


im2 = im.resize((60, 40)).convert("L")


print(im2)


import numpy as np


print(np.array(im2))


np.array(im2).max()


im2



