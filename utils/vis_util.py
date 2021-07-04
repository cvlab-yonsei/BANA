import numpy as np
from PIL import Image

CLASS_COLORS = (
    (0, 0, 0),
    (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), 
    (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), 
    (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), 
    (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), 
    (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
    (224, 224, 192)
)


def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)
    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3
        cmap[i] = np.array([r, g, b])
    cmap = cmap/255 if normalized else cmap
    return cmap


def decode_VOC_labels(mask):
    num_classes = len(CLASS_COLORS)
    h, w = mask.shape
    img = Image.new('RGB', (w, h))
    pixels = img.load()
    for j_, j in enumerate(mask):
        for k_, k in enumerate(j):
            if k < num_classes:
                pixels[k_, j_] = CLASS_COLORS[k]
            if k == 255:
                pixels[k_, j_] = CLASS_COLORS[-1]
    output = np.array(img)
    return output


def get_npimg(x):
    img = x.clone()
    img -= img.min()
    img /= img.max()
    img *= 255
    img = img.permute(1,2,0).numpy().astype('int')
    img = img[:,:,::-1]
    return img