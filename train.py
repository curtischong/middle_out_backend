
#crop image to middle
image_width = 500
image_height = 500
input_path = "input_photos"

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(input_path) if isfile(join(input_path, f))]


#as
