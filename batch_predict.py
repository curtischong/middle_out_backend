import numpy as np
import os
import re
import tensorflow as tf
from voxel_flow_model import Voxel_flow_model
from utils.image_utils import imwrite
from utils.image_utils import imwrite_better
import sys
from datetime import datetime

IMG_WIDTH = 256
IMG_HEIGHT = 256

np.set_printoptions(threshold=np.nan)


FLAGS = tf.app.flags.FLAGS

# Define necessary FLAGS
tf.app.flags.DEFINE_string('train_dir', './voxel_flow_checkpoints/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_image_dir', './voxel_flow_train_image/',
			   """Directory where to output images.""")
tf.app.flags.DEFINE_string('test_image_dir', './voxel_flow_test_image/',
			   """Directory where to output images.""")
tf.app.flags.DEFINE_string('subset', 'train',
                           """Either 'train' or 'validation'.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', 'checkpoints/voxel_flow_checkpoints',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer(
        'batch_size', 3, 'The number of samples in each batch.')
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0003,
                          """Initial learning rate.""")

#crop image to middle
image_width = 500
image_height = 500
input_path = "input_photos"

from os import listdir
from os.path import isfile, join


from PIL import Image
from numpy import*


def newFrame(cur_filepath):
    # Convert image to black and white
    img = Image.open(cur_filepath).convert('1')
    # Pad image with whitespace
    max_size = max(img.size[0], img.size[1])
    new_size = (max_size, max_size)
    padded = Image.new('1', new_size, 255)
    padded.paste(img, ((new_size[0]-img.size[0])//2, (new_size[1]-img.size[1])//2))
    # Scale image to 256/256
    padded.thumbnail((256,256))
    return array(padded.getdata(), np.uint8).reshape(256, 256, 1)

x1 = []
x2 = []
path = "input_frames"
cur_frames = []
included_extensions = ['jpg', 'bmp', 'png', 'gif']
onlyfiles = [fn for fn in os.listdir(path)
                if any(fn.endswith(ext) for ext in included_extensions)]
for one_file in onlyfiles:
    cur_frames.append(newFrame(path + "/" + one_file))

for idx in range(len(cur_frames) - 1):
    x1.append(cur_frames[idx])
    x2.append(cur_frames[idx + 1])


final_arr_of_frames = []


def predict1():
  """Perform test on a trained model."""
  with tf.Graph().as_default():
		# Create input and target placeholder.
    input_placeholder = tf.placeholder(tf.float32, shape=(3, 256, 256, 2))
    target_placeholder = tf.placeholder(tf.float32, shape=(3, 256, 256, 1))

    # input_resized = tf.image.resize_area(input_placeholder, [128, 128])
    # target_resized = tf.image.resize_area(target_placeholder,[128, 128])

    # Prepare model.
    model = Voxel_flow_model(is_train=True)
    prediction = model.inference(input_placeholder)
    # reproduction_loss, prior_loss = model.loss(prediction, target_placeholder)
    reproduction_loss = model.loss(prediction, target_placeholder)
    # total_loss = reproduction_loss + prior_loss
    total_loss = reproduction_loss

    # Create a saver and load.
    saver = tf.train.Saver(tf.all_variables())
    sess = tf.Session()

    # Restore checkpoint from file.
    if FLAGS.pretrained_model_checkpoint_path:
      assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
      ckpt = tf.train.get_checkpoint_state(
               FLAGS.pretrained_model_checkpoint_path)
      restorer = tf.train.Saver()
      restorer.restore(sess, ckpt.model_checkpoint_path)
      print('%s: Pre-trained model restored from %s' %
        (datetime.now(), ckpt.model_checkpoint_path))

    # Process on test dataset.
    data_list_frame1 = curx1
    data_size = len(data_list_frame1)
    epoch_num = int(data_size / FLAGS.batch_size)

    data_list_frame2 = curx2
    data_list_frame3 = curx2

    # Load single data.
    line_image_frame1 = np.array([data_list_frame1,data_list_frame1,data_list_frame1])
    line_image_frame2 = np.array([data_list_frame2,data_list_frame2,data_list_frame2])
    line_image_frame3 = np.array([data_list_frame3,data_list_frame3,data_list_frame3])

    print(line_image_frame1.shape)

    feed_dict = {input_placeholder: np.concatenate((line_image_frame1, line_image_frame3), 3),
                target_placeholder: line_image_frame2}
    # Run single step update.
    prediction_np, target_np, loss_value = sess.run([prediction,
                                                    target_placeholder,
                                                    total_loss],
                                                    feed_dict = feed_dict)
    final_arr_of_frames.append(prediction_np[-1,:,:,:])
    #file_name_label = FLAGS.test_image_dir+str(i)+'_gt.png'
    #imwrite(file_name, prediction_np[-1,:,:,:])
    #imwrite(file_name_label, target_np[-1,:,:,:])

x1 = []
x2 = []

while(len(l1) > 0):
    curx1 = x1[0:3]
    curx2 = x2[0:3]
    x1 = x1[3:]
    x2 = x2[3:]
    predict()

for idx in range(len(final_arr_of_frames)):
    mwrite("style_input/out" + str(idx) + ".png", final_arr_of_frames[idx])
