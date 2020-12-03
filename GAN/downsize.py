from pathlib import Path
import os
from PIL import Image

size = 240, 135

for path in Path('data/images').rglob('*.jpg'):
    outfile = "{0}{1}".format("data/small_images/", path.name)
    print(outfile)
    im = Image.open(path)
    im.thumbnail(size, Image.ANTIALIAS)
    im.save(outfile, "JPEG")
