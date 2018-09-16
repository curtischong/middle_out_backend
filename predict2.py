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



"""with tf.Graph().as_default():
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
    #saver = tf.train.import_meta_graph('checkpoints/voxel_flow_checkpoints/model.ckpt-130000.meta')

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge_all()

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()
    sess = tf.Session()

    #saver = tf.train.import_meta_graph("checkpoints/voxel_flow_checkpoints/model.ckpt-130000.meta")
    #saver.restore(sess, tf.train.latest_checkpoint('checkpoints/voxel_flow_checkpoints/'))

    # Restore checkpoint from file.
    if FLAGS.pretrained_model_checkpoint_path:
      assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
      ckpt = tf.train.get_checkpoint_state(
               FLAGS.pretrained_model_checkpoint_path)
      restorer = tf.train.Saver()
      restorer.restore(sess, ckpt.model_checkpoint_path)
      print('%s: Pre-trained model restored from %s' %
        (datetime.now(), ckpt.model_checkpoint_path))

    sess.run(init)

    # Summary Writter
    summary_writer = tf.summary.FileWriter(
        FLAGS.train_dir,
        graph=sess.graph)"""

"""p = np.random.permutation(len(x1))
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
        saver.save(sess, checkpoint_path, global_step=step)"""



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

pic1 = newFrame("two_input_files/person1.png")
pic2 = newFrame("two_input_files/person2.png")
pic3 = newFrame("two_input_files/person2.png")



def test():
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
    data_list_frame1 = pic1
    data_size = len(data_list_frame1)
    epoch_num = int(data_size / FLAGS.batch_size)

    data_list_frame2 = pic2
    data_list_frame3 = pic2

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

    file_name = "two_input_files/out.png"
    #file_name_label = FLAGS.test_image_dir+str(i)+'_gt.png'
    imwrite(file_name, prediction_np[-1,:,:,:])
    #imwrite(file_name_label, target_np[-1,:,:,:])

test()