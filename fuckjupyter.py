import numpy as np
import os
import re
import tensorflow as tf
from voxel_flow_model import Voxel_flow_model
from utils.image_utils import imwrite
from utils.image_utils import imwrite_better
import sys

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
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', './voxel_flow_checkpoints/',
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

included_extensions = ['jpg', 'bmp', 'png', 'gif']
onlyfiles = [fn for fn in os.listdir(input_path)
              if any(fn.endswith(ext) for ext in included_extensions)]

onlyfiles[:2]

from PIL import Image
from numpy import*



# crop to center of image
# I might want an assert statement to make sure that the dimensions fit the input.
#note: I think the input of the data is actually in the wrong orientation. it should be 720,1080
#offset_left = int((frames.shape[1] - IMG_WIDTH)/2)
#offset_top = int((frames.shape[2] - IMG_HEIGHT)/2)
#frames = frames[:,offset_left : (offset_left + IMG_WIDTH), offset_top : (offset_top + IMG_HEIGHT), :]
#Image.fromarray(frames[0].reshape([256, 256])).show()
#sys.exit(1)

#REMEMBER TO CHANGE THE imwrite function!!!!!!

"""for i in range(len(frames)):
    file_name = "input_pics/"+str(i)+'_input.png'
    print(frames[i, :, :, :].shape)
    imwrite(file_name, frames[i, :, :, :])"""

frames = np.load('frames.npy')

print("frame shape:")
print(frames.shape)


def flip_frames(frame):
    frames_horiz = np.fliplr(frame)
    frames_vert = np.flipud(frame)
    frames_horiz_vert = np.flipud(np.fliplr(frame))

    return (frame, frames_horiz, frames_vert, frames_horiz_vert)


target = []
x1 = []
x2 = []
def middle_out(frame1, frame2, frame3):
    a,b,c,d = flip_frames(frame2)
    target.append(a)
    target.append(b)
    target.append(c)
    target.append(d)

    a1,b1,c1,d1 = flip_frames(frame1)
    x1.append(a1)
    x1.append(b1)
    x1.append(c1)
    x1.append(d1)

    a2,b2,c2,d2 = flip_frames(frame3)
    x2.append(a2)
    x2.append(b2)
    x2.append(c2)
    x2.append(d2)


def pick_out_frames(cur_scene):
    for idx in range(0,len(cur_scene)-2):
        middle_out(cur_scene[idx], cur_scene[idx + 1], cur_scene[idx + 2])


for dir in frames:
    pick_out_frames(dir)

target = np.array(target)
x1 = np.array(x1)
x2 = np.array(x2)

print("frame shape:", target.shape)





with tf.Graph().as_default():
    # Create input and target placeholder.
    input_placeholder = tf.placeholder(tf.float32, shape=(3, 256, 256, 2))
    target_placeholder = tf.placeholder(tf.float32, shape=(3, 256, 256, 1))

    # input_resized = tf.image.resize_area(input_placeholder, [128, 128])
    # target_resized = tf.image.resize_area(target_placeholder,[128, 128])

    # Prepare model.
    model = Voxel_flow_model()
    prediction = model.inference(input_placeholder)
    # reproduction_loss, prior_loss = model.loss(prediction, target_placeholder)
    reproduction_loss = model.loss(prediction, target_placeholder)
    # total_loss = reproduction_loss + prior_loss
    total_loss = reproduction_loss

    # Perform learning rate scheduling.
    learning_rate = FLAGS.initial_learning_rate

    # Create an optimizer that performs gradient descent.
    opt = tf.train.AdamOptimizer(learning_rate)
    grads = opt.compute_gradients(total_loss)
    update_op = opt.apply_gradients(grads)

    # Create summaries
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    summaries.append(tf.summary.scalar('total_loss', total_loss))
    summaries.append(tf.summary.scalar('reproduction_loss', reproduction_loss))
    # summaries.append(tf.scalar_summary('prior_loss', prior_loss))
    summaries.append(tf.summary.image('Input Image', input_placeholder, 3))
    summaries.append(tf.summary.image('Output Image', prediction, 3))
    summaries.append(tf.summary.image('Target Image', target_placeholder, 3))

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge_all()

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Summary Writter
    summary_writer = tf.summary.FileWriter(
        FLAGS.train_dir,
        graph=sess.graph)

    p = np.random.permutation(len(x1))
    data_list_frame1 = x1[p]#np.expand_dims(x1[p], axis=3)
    data_list_frame2 = target[p]#np.expand_dims(target[p], axis=3)
    data_list_frame3 = x2[p]#np.expand_dims(x2[p], axis=3)

    data_size = len(data_list_frame1)
    epoch_num = int(data_size / FLAGS.batch_size)


    print("starting training")

    for step in range(0, FLAGS.max_steps):
      batch_idx = step % epoch_num

      batch_data_list_frame1 = data_list_frame1[int(
          batch_idx * FLAGS.batch_size): int((batch_idx + 1) * FLAGS.batch_size)]
      batch_data_list_frame2 = data_list_frame2[int(
          batch_idx * FLAGS.batch_size): int((batch_idx + 1) * FLAGS.batch_size)]
      batch_data_list_frame3 = data_list_frame3[int(
          batch_idx * FLAGS.batch_size): int((batch_idx + 1) * FLAGS.batch_size)]

      # Load batch data.
      feed_dict = {input_placeholder: np.concatenate(
          (batch_data_list_frame1, batch_data_list_frame3), 3), target_placeholder: batch_data_list_frame2}


      # Run single step update.
      _, loss_value = sess.run([update_op, total_loss], feed_dict=feed_dict)

      if batch_idx == 0: # this is bad shuffling code. I should do it at the end of each epoch :(
        # Shuffle data at each epoch.
        #random.seed(1)
        p = np.random.permutation(len(x1))
        data_list_frame1 = x1[p]#np.expand_dims(x1[p], axis=3)
        data_list_frame2 = target[p]#np.expand_dims(target[p], axis=3)
        data_list_frame3 = x2[p]#np.expand_dims(x2[p], axis=3)
        print('Epoch Number: %d' % int(step / epoch_num))

      # Output Summary
      if step % 10 == 0:
        # summary_str = sess.run(summary_op, feed_dict = feed_dict)
        # summary_writer.add_summary(summary_str, step)
	      print("Loss at step %d: %f" % (step, loss_value))

      if step % 500 == 0:
        # Run a batch of images
        prediction_np, target_np = sess.run(
            [prediction, target_placeholder], feed_dict=feed_dict)
        for i in range(0, prediction_np.shape[0]):
          file_name = FLAGS.train_image_dir+str(i)+'_out_bad.png'
          file_name_label = FLAGS.train_image_dir+str(i)+'_gt.png'
          imwrite(file_name, prediction_np[i, :, :, :])
          imwrite(file_name_label, target_np[i, :, :, :])

          file_name_better = FLAGS.train_image_dir+str(i)+'_out_bad.png'
          imwrite_better(file_name_better, target_np[i, :, :, :])


      # Save checkpoint
      if step % 5000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

