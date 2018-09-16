import numpy as np
from numpy import *
import os
import re
import tensorflow as tf
from voxel_flow_model import Voxel_flow_model
from utils.image_utils import imwrite
import sys
from PIL import Image


def newFrame(cur_filepath):
    # Convert image to black and white
    img = Image.open(cur_filepath).convert('1')
    # Pad image with whitespace
    max_size = max(img.size[0], img.size[1])
    new_size = (max_size, max_size)
    padded = Image.new('1', new_size, 512)
    padded.paste(img, ((new_size[0]-img.size[0])//2, (new_size[1]-img.size[1])//2))
    # Scale image to 256/256
    padded.thumbnail((512,512))
    return array(padded.getdata(), np.uint8).reshape(512, 512, 1)


frames = []

subdirectories = [x[0] for x in os.walk("OUTPUT")]
print("number of subdirectories: ", len(subdirectories))
for path in subdirectories[1:]:
    cur_frames = []
    included_extensions = ['jpg', 'bmp', 'png', 'gif']
    onlyfiles = [fn for fn in os.listdir(path)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    for one_file in onlyfiles:
        try:
            the_frame = newFrame(path + "/" + one_file)
            cur_frames.append(the_frame)
        except:
            print("failed " + str(path))
    print("finished reading directory: " + path)
    frames.append(cur_frames)

import json

with open('data.json', 'w') as outfile:
    json.dump(frames, outfile)

#np.save("frames.npy", frames)