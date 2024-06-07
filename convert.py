import os
import shutil
import numpy as np
import imageio.v2 as imageio

def main_train():
    path = r'C:\Users\jiahao\Downloads\mnist_png-master\mnist_png-master\mnist_png\mnist_png\training'
    files = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if '.png' in fname:
                if 0 <= float(fname.replace('.png', '')) < 10000:
                    files.append(os.path.join(dirpath, fname))

    path = r'D:\pySimpleDenoise\minist_dataset\train'

    for idx, item in enumerate(files):
        # img = imageio.imread(item)
        # img = img.astype(np.float32)
        # img = np.mean(img, 0)
        # img = img.round().clip(0, 255).astype(np.uint8)
        # imageio.imwrite(os.path.join(path, '{:0>4}.png'.format(idx)), img)
        shutil.copy(item, os.path.join(path, '{:0>4}.png'.format(idx)))


def main_val():
    path = r'C:\Users\jiahao\Downloads\mnist_png-master\mnist_png-master\mnist_png\mnist_png\training'
    files = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if '.png' in fname:
                if 10000 <= float(fname.replace('.png', '')) < 10100:
                    files.append(os.path.join(dirpath, fname))

    path = r'D:\pySimpleDenoise\minist_dataset\val'

    for idx, item in enumerate(files):
        # img = imageio.imread(item)
        # img = img.astype(np.float32)
        # img = np.mean(img, 0)
        # img = img.round().clip(0, 255).astype(np.uint8)
        # imageio.imwrite(os.path.join(path, '{:0>4}.png'.format(idx)), img)
        shutil.copy(item, os.path.join(path, '{:0>4}.png'.format(idx)))


def main_test():
    path = r'C:\Users\jiahao\Downloads\mnist_png-master\mnist_png-master\mnist_png\mnist_png\training'
    files = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if '.png' in fname:
                if 10100 <= float(fname.replace('.png', '')) < 10200:
                    files.append(os.path.join(dirpath, fname))

    path = r'D:\pySimpleDenoise\minist_dataset\test'

    for idx, item in enumerate(files):
        # img = imageio.imread(item)
        # img = img.astype(np.float32)
        # img = np.mean(img, 0)
        # img = img.round().clip(0, 255).astype(np.uint8)
        # imageio.imwrite(os.path.join(path, '{:0>4}.png'.format(idx)), img)
        shutil.copy(item, os.path.join(path, '{:0>4}.png'.format(idx)))

if __name__ == '__main__':
    main_train()
    main_val()
    main_test()