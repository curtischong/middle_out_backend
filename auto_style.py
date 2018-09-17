import subprocess
import os

path = "style_input"
cur_frames = []
included_extensions = ['jpg', 'bmp', 'png', 'gif']
onlyfiles = [fn for fn in os.listdir(path)
                if any(fn.endswith(ext) for ext in included_extensions)]
for one_file in onlyfiles:
    subprocess.call(["python3", "neural_style.py", "--content", "style_input/" + one_file, "--styles", "examples/2-style1.jpg", "--output", "style_output/" + one_file, "--overwrite", "--iterations", "100"])