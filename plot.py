import cPickle
import hickle as hkl
import os
import pdb
import numpy as np
from PIL import Image as Im
import sys

expand_factor = 300
nsamples = 12500

# retrieve the first nsamples images
dim = 255 * 255 * 3
nsplits = np.floor(nsamples / 256)
nsamples = nsplits * 256

img = np.zeros((nsamples, dim), dtype=np.uint8)
data_folder = "/mnt/bigdisk/data/datasets/gender-1M/valid/"
for i, filename in enumerate(sorted(os.listdir(data_folder))):
    x = hkl.load(os.path.join(data_folder, filename))
    x = np.swapaxes(x, 0, 3)
    x = x.reshape((256, dim))
    img[i * 256:(i + 1) * 256, :] = x
    if i >= nsplits - 1: break
print nsplits, "nsplits"

print "loading tsne coordinates..."
tsne = cPickle.load(open("tsne-representation.pkl"))[:nsamples]
tsne -= tsne.min()
tsne *= expand_factor

size = np.ceil(tsne.max()) + 300
print "printing poster size : (%i, %i)" % (size, size)
# poster
bigpic = np.ones((size, size, 3), dtype="uint8") * 255
for k, (im, coord) in enumerate(zip(img, tsne)):
    print "%.2f %% completed\r" %((k + 1) * 100. / nsamples),
    sys.stdout.flush()
    x, y = int(coord[0]), int(coord[1])
    im = Im.fromarray(im.reshape((255, 255, 3))).resize((50, 50), Im.ANTIALIAS)
    bigpic[x:x + 50, y:y + 50, :] = np.array(im).astype("uint8")

Im.fromarray(bigpic).convert('RGB').save("tsne.jpg")
