# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import re


TOTAL_EXAMPLES_PER_EPOCH_TRAINING = 200
TOTAL_EXAMPLES_PER_EPOCH_VALIDATION = 100
TOTAL_EXAMPLES_PER_EPOCH_TESTING = 180

HEIGHT = 184
WIDTH = 184
MAX_STEPS = 5


def getFileList(file_dir):
    name_list = []
    for im in os.listdir(file_dir):
        name_list.append(file_dir + im)
    return name_list


def GetFileNameList(file_dir):
    """获取in和out的图像名称列表"""
    with tf.name_scope('FileNameList'):
        pictureRaw = []
        pictureSeg = []
        for im in os.listdir(file_dir):
            # all_file
            m = re.match("(.{30,36})(_o\\.png)", im)
            if m:
                # pictureRaw.append(im)
                # pictureSeg.append(m.group(1) + '_o.png')
                pictureSeg.append(file_dir+im)
                pictureRaw.append(file_dir+m.group(1) + '.png')
    return pictureRaw, pictureSeg


def GetBatchFromFile_Train(rawImageList, segImageList, BatchSize):
    '''
    Args:
        rawDir: Directory of raw and segmante images
        BatchSize: batch size
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 1], dtype=tf.float32
        label_batch: 4D tensor [batch_size, width, height, 1], dtype=tf.float32
    '''
    #rawImageList, segImageList = GetFileNameList(rawDir)

    rawImageList = tf.cast(rawImageList, tf.string, name='CastrawFileName')
    segImageList = tf.cast(segImageList, tf.string, name='CastsegFileName')

    assert rawImageList.shape[0].value == segImageList.shape[0].value, 'Dimension Error --> NameList'
    NUM_EXAMPLES = rawImageList.shape[0].value
    print("Total Validation Examples: %d" % NUM_EXAMPLES)

    # Make an input queue
    InputQueue = tf.train.slice_input_producer([rawImageList, segImageList],
                                               num_epochs=None,
                                               shuffle=False,
                                               capacity=16,
                                               shared_name=None,
                                               name='SliceInputProducer')

    # Read one example from input queue
    rawImageContent = tf.read_file(InputQueue[0], name='ReadrawImage')
    segImageContent = tf.read_file(InputQueue[1], name='ReadsegImage')

    # Decode the jpeg image format
    rawImage = tf.image.decode_image(rawImageContent, channels=1, name='DecodeRawImage')
    segImage = tf.image.decode_image(segImageContent, channels=1, name='DecodeSegImage')

    # with tf.name_scope('Random_cut'):
    #     GIBBS_y = tf.random_uniform(shape = (), minval = 0, maxval = GIBBS_HEIGHT - GIBBS_HEIGHT_PART - 1, dtype = tf.int32)
    #     GIBBS_x = tf.random_uniform(shape = (), minval = 0, maxval = GIBBS_WIDTH - GIBBS_WIDTH_PART - 1, dtype = tf.int32)
    #     CLEAR_y =  GIBBS_y * SCALE
    #     CLEAR_x =  GIBBS_x * SCALE
    #     GIBBSImage = GIBBSImage[GIBBS_x: GIBBS_x + GIBBS_WIDTH_PART, GIBBS_y : GIBBS_y + GIBBS_HEIGHT_PART,:]
    #     CLEARImage = CLEARImage[CLEAR_x: CLEAR_x + CLEAR_WIDTH_PART, CLEAR_y : CLEAR_y + CLEAR_HEIGHT_PART,:]
    #
    # with tf.name_scope('SetShape'):
    #     GIBBSImage.set_shape([GIBBS_WIDTH_PART, GIBBS_HEIGHT_PART, 1])
    #     CLEARImage.set_shape([CLEAR_WIDTH_PART, CLEAR_HEIGHT_PART, 1])

    # with tf.name_scope('augmentation'):
    #     GIBBSImage, CLEARImage = augmentation(GIBBSImage, CLEARImage)

    # Set the min fraction of examples that will enqueue

    # MIN_FRACTION_EXAMPLES_IN_QUEUE = 0.05
    # MIN_EXAMPLES_IN_QUEUE = int(NUM_EXAMPLES * MIN_FRACTION_EXAMPLES_IN_QUEUE)
    # print ('Filling queue with %d/%d images. This will take a few minutes.' %
    #        (MIN_EXAMPLES_IN_QUEUE, NUM_EXAMPLES))

    with tf.name_scope('SetShape'):
        rawImage.set_shape([WIDTH, HEIGHT, 1])
        segImage.set_shape([WIDTH, HEIGHT, 1])
    rawImageBatch, segImageBatch = tf.train.shuffle_batch([rawImage, segImage],
                                                          batch_size=BatchSize,
                                                          num_threads=8,
                                                          capacity=10 * BatchSize,
                                                          min_after_dequeue=2 * BatchSize,
                                                          name='SuffleBatch')

    tf.summary.image('train_GIBBS_images', rawImageBatch, max_outputs=4)
    tf.summary.image('train_CLEAR_images', segImageBatch, max_outputs=4)

    # Cast to tf.float32
    with tf.name_scope('CastToFloat'):
        rawImageBatch = tf.cast(rawImageBatch, tf.float32)
        segImageBatch = tf.cast(segImageBatch, tf.float32)

    # Normalization
    with tf.name_scope('Normalization'):
        rawImageBatch = rawImageBatch / 255.0
        segImageBatch = segImageBatch / 255.0

    return rawImageBatch, segImageBatch


def GetBatchFromFile_Valid(GIBBSDir, CLEARDir, BatchSize):
    ''' Get batch from files for validation.
    Args:
        GIBBSDir: Directory of GIBBS images
        CLEARDir: Directory of CLEAR images
        BatchSize: batch size
    Returns:
        image_batch: 4D tensor [batch_size, height, width, 1], dtype=tf.float32
        label_batch: 4D tensor [batch_size, height, width, 1], dtype=tf.float32
    '''
    GIBBSImageList = tf.cast(GIBBSDir, tf.string, name = 'CastGIBBSFileName')
    CLEARImageList = tf.cast(CLEARDir, tf.string, name = 'CastCLEARFileName')
    if GIBBSImageList.shape[0].value != CLEARImageList.shape[0].value:
        print('Dimension Error --> NameList')
        return
    
    NUM_EXAMPLES = GIBBSImageList.shape[0].value
    print("Total Training Examples: %d" % NUM_EXAMPLES)
    
    InputQueue = tf.train.slice_input_producer([GIBBSImageList, CLEARImageList, CLEARImageList],
                                               num_epochs = None, # validate once or not?
                                               shuffle = False,   # DO NOT shuffle!
                                               capacity = 32,
                                               shared_name = None,
                                               name = 'StringInputProducer')
    
    # Read one example from input queue

    GIBBSImageContent = tf.read_file(InputQueue[0], name = 'ReadGIBBSImage')
    CLEARImageContent = tf.read_file(InputQueue[1], name = 'ReadCLEARImage')
    name = InputQueue[2]
    # Decode the png image format
    GIBBSImage = tf.image.decode_image(GIBBSImageContent, channels = 1, name = 'DecodeGIBBSImage')
    CLEARImage = tf.image.decode_image(CLEARImageContent, channels = 1, name = 'DecodeCLEARImage')
    
    MIN_FRACTION_EXAMPLE_IN_QUEUE = 0.05
    MIN_EXAMPLES_IN_QUEUE = int(NUM_EXAMPLES * MIN_FRACTION_EXAMPLE_IN_QUEUE)
    print ('Filling queue with %d/%d images. This will take a few minutes.' % 
           (MIN_EXAMPLES_IN_QUEUE, NUM_EXAMPLES))
    
    # Set shape for images
    with tf.name_scope('SetShape'):
        GIBBSImage.set_shape([None, None, 1])
        CLEARImage.set_shape([None, None, 1])
    
    GIBBSImageBatch, CLEARImageBatch, name = tf.train.batch([GIBBSImage, CLEARImage, name],
                                                batch_size = BatchSize,
                                                num_threads = 1, # set to 1 to keep order.
                                                capacity = MIN_EXAMPLES_IN_QUEUE + 10*BatchSize,
                                                dynamic_pad = True,
                                                name = 'Batch')
    tf.summary.image('val_GIBBS_images', GIBBSImageBatch, max_outputs = 4)
    tf.summary.image('val_CLEAR_images', CLEARImageBatch, max_outputs = 4)
    
    # Cast to tf.float32
    with tf.name_scope('CastToFloat'):
        GIBBSImageBatch = tf.cast(GIBBSImageBatch, tf.float32)
        CLEARImageBatch = tf.cast(CLEARImageBatch, tf.float32)
    
    # Normalization
    with tf.name_scope('Normalization'):
        GIBBSImageBatch = GIBBSImageBatch / 255.0
        CLEARImageBatch = CLEARImageBatch / 255.0
    
    return GIBBSImageBatch, CLEARImageBatch


'''
def augmentation(input_patch, label_patch):
    """ Data Augmentation with TensorFlow ops.
    
    Args:
        input_patch: input tensor representing an input patch or image
        label_patch: label tensor representing an target patch or image
    Returns:
        rotated input_patch and label_patch randomly
    """
    def no_trans():
        return input_patch, label_patch

    def vflip():
        inpPatch = input_patch[::-1, :, :]
        labPatch = label_patch[::-1, :, :]
        return inpPatch, labPatch
    
    def hflip():
        inpPatch = input_patch[:, ::-1, :]
        labPatch = label_patch[:, ::-1, :]
        return inpPatch, labPatch
    
    def hvflip():
        inpPatch = input_patch[::-1, ::-1, :]
        labPatch = label_patch[::-1, ::-1, :]
        return inpPatch, labPatch

    def trans():
        inpPatch = tf.image.transpose_image(input_patch[:, :, :])
        labPatch = tf.image.transpose_image(label_patch[:, :, :])
        return inpPatch, labPatch
    
    def tran_vflip():
        inpPatch = tf.image.transpose_image(input_patch)[::-1, :, :]
        labPatch = tf.image.transpose_image(label_patch)[::-1, :, :]
        return inpPatch, labPatch
    
    def tran_hflip():
        inpPatch = tf.image.transpose_image(input_patch)[:, ::-1, :]
        labPatch = tf.image.transpose_image(label_patch)[:, ::-1, :]
        return inpPatch, labPatch
        
    def tran_hvflip():
        inpPatch = tf.image.transpose_image(input_patch)[::-1, ::-1, :]
        labPatch = tf.image.transpose_image(label_patch)[::-1, ::-1, :]
        return inpPatch, labPatch
    
    rot = tf.random_uniform(shape = (), minval = 2, maxval = 9, dtype = tf.int32)    
    input_patch, label_patch = tf.case({tf.equal(rot, 2): vflip,
                                  tf.equal(rot, 3): hflip,
                                  tf.equal(rot, 4): hvflip,
                                  tf.equal(rot, 5): trans,
                                  tf.equal(rot, 6): tran_vflip,
                                  tf.equal(rot, 7): tran_hflip,
                                  tf.equal(rot, 8): tran_hvflip},
    default = no_trans, exclusive = True)
    
    return input_patch, label_patch
'''

if __name__ == '__main__':

    file_dir = 'D:\\OneDrive\\kaggleP\\3rd\\kaggle_ndsb2-master\\data_validate'
    rawList, segList = GetFileNameList(file_dir)
    rawBatch, segBatch = GetBatchFromFile_Train(rawList, segList, 2)
    
    step = 0

    with tf.Session() as sess:
        batch_index = 0
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop():
                inp_batch, label_batch = sess.run([rawBatch, segBatch])
                #            label_batch = sess.run(labelBatch)
                for i in range(2):
                    image_name = rawList[i].split('/')[-1]
                    #plt.imshow(inp_batch[0, :, :, 0], cmap=plt.cm.gray)
                    #plt.title('input batch%d: image %d' % (batch_index, i))
                    #plt.show()
                    image_name = segList[i].split('/')[-1]
                  #  plt.imshow(label_batch[0, :, :, 0], cmap=plt.cm.gray)
                    #plt.title('input batch%d: image %d' % (batch_index, i))
                    #plt.show()
                    print(" --------------------------------------------------------")
                batch_index += 1
        except tf.errors.OutOfRangeError:
            print("Done!")
        finally:
            coord.request_stop()
            coord.join()