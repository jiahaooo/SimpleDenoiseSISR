import imageio.v2
import numpy as np

url = r'D:\pySimpleDenoiseSISR\minist_dataset\val\0000.png'

img = imageio.v2.imread(url)
print(img.shape) # 输出图像尺寸
print(img.dtype) # 输出图像数据类型
img = img.reshape(1, 1, 28, 28) # 将图像按照Tensor格式化
img = img.astype(np.float32) # 将图像的数据类型转换为32bit浮点数
print('\n')
print(img.shape) # 输出图像尺寸
print(img.dtype) # 输出图像数据类型