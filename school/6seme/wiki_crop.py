import os
import glob
from PIL import Image

dst_dir = './data/resize'
os.makedirs(dst_dir, exist_ok=True)
files = glob.glob('./data/*.jpg')

for f in files:
    img = Image.open(f)
    img_resize = img.resize((299, 299))
    root, ext = os.path.splitext(f)
    basename = os.path.basename(root)
    img_resize.save(os.path.join(dst_dir, basename + '_resize' + ext))

