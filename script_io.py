import os

path = r'D:\pySimpleDenoiseSISR\minist_dataset\val'
list_img = []
for item in os.listdir(path):
    if '.png' in item:
        print(os.path.join(path, item))
        list_img.append(os.path.join(path, item))
print(list_img.__len__())