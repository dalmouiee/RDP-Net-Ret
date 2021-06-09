'''
    Author: Daniel Al Mouiee
    Date:   09/06/2021

    Script to convert images from TIF to JPEG format
'''

import os
from PIL import Image

yourpath = os.getcwd()
for root, dirs, files in os.walk(yourpath, topdown=False):
    print(root,", ", dirs,", ", files ,", ")
    for name in files:
        print(os.path.join(root, name))
        if os.path.splitext(os.path.join(root, name))[1].lower() == ".tif":
            if os.path.isfile(os.path.splitext(os.path.join(root, name))[0] + ".jpg"):
                print("A jpeg file already exists for %s" % name)
            # If a jpeg is *NOT* present, create one from the tiff.
            else:
                outfile = os.path.splitext(os.path.join(root, name))[0] + ".jpg"
                im = Image.open(os.path.join(root, name))
                print("Generating jpeg for %s" % name)
                im.thumbnail(im.size)
                im.save(outfile, "JPEG", quality=100)