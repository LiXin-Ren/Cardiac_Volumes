from __future__ import absolute_import

from datetime import datetime
import tensorflow as tf
import unet_input

import unet_model
import settings_unet

import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from scipy import misc


parser = settings_unet.parser
PARAMS = parser.parse_args()
SAVE_IMAGE_DIR = "../predict_volumns/seged_images"
 
HEIGHT = 256
WIDTH = 256

file_dir = "../preprocess/preprocessed_images/"

rawNameList = unet_input.getFileList(file_dir)
TOTAL_VALID_IMAGES = len(rawNameList)



def SegAndSaveImages(saver, raw_image, seg_image):
    with tf.Session(config=tf.ConfigProto(log_device_placement=PARAMS.log_device_placement)) as sess:
        ckpt = tf.train.get_checkpoint_state(PARAMS.log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            #global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('no checkpoint file found!')
            return

        # start the queue runners
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            # calculate the number of iterations for validation set. Because the batch size in this case
            # is 1, we just need to excute TOTAL_VALID_EXAMPLES times loop.
            step = 0
            while not coord.should_stop() and step < TOTAL_VALID_IMAGES:
                print('processing image %d/%d...' % (step + 1, TOTAL_VALID_IMAGES))

                image_raw, image_seg = sess.run([raw_image, seg_image])

                image_name = rawNameList[step].split('/')[-1]
                image_path = os.path.join(SAVE_IMAGE_DIR, image_name)

                misc.imsave(image_path, image_seg[0, :, :, 0])

                step += 1


        except Exception as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)





def GetBatch(rawImageList, BatchSize = 1):
    rawImageList = tf.cast(rawImageList, tf.string, name='CastrawFileName')

    NUM_EXAMPLES = rawImageList.shape[0].value
    print("Total Validation Examples: %d" % NUM_EXAMPLES)

    # Make an input queue
    InputQueue = tf.train.slice_input_producer([rawImageList],
                                               num_epochs=None,
                                               shuffle=False,
                                               capacity=16,
                                               shared_name=None,
                                               name='SliceInputProducer')

    # Read one example from input queue
    rawImageContent = tf.read_file(InputQueue[0], name='ReadrawImage')


    # Decode the jpeg image format
    rawImage = tf.image.decode_image(rawImageContent, channels=1, name='DecodeRawImage')

    with tf.name_scope('SetShape'):
        rawImage.set_shape([WIDTH, HEIGHT, 1])

    rawImageBatch = tf.train.batch([rawImage],
                                                  batch_size=BatchSize,
                                                  name='SuffleBatch')


    # Cast to tf.float32
    with tf.name_scope('CastToFloat'):
        rawImageBatch = tf.cast(rawImageBatch, tf.float32)

    # Normalization
    with tf.name_scope('Normalization'):
        rawImageBatch = rawImageBatch / 255.0

    return rawImageBatch

def segmentation(file_dir):
    with tf.Graph().as_default() as g:
        #        rawNameList, segNameList = unet_input.GetFileNameList(VAL_DIR)
        #       TOTAL_VALID_IMAGES = len(rawNameList)

        raw_image = GetBatch(rawNameList)

        # Build computational graph
        seg_image = unet_model.UNet(raw_image)
       
        saver = tf.train.Saver()
        #summ_op = tf.summary.merge_all()
        #summ_writer = tf.summary.FileWriter(PARAMS.val_log_dir, g)
        while True:
            SegAndSaveImages(saver, raw_image, seg_image)
            break


if __name__ == '__main__':
   # tf.app.run()
    segmentation(file_dir)

