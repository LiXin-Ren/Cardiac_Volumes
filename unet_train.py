# -*- coding: utf-8 -*-
from __future__ import absolute_import
from datetime import datetime
import tensorflow as tf
#import settings
import unet_input
import settings_unet
import time
import os
import unet_model
#import RVseg_model
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Some basic constants for the model, supporting command line parameters
parser = settings_unet.parser
PARAMS = parser.parse_args()
# Parse command line parameters
BATCH_SIZE = PARAMS.batch_size
LearningRate = PARAMS.learning_rate

LogFreq = PARAMS.log_freq
LogDir = PARAMS.log_dir

AllowSoftPlacement = PARAMS.allow_soft_placement
TrainFromExist = PARAMS.train_from_exist
ExistModelDir = PARAMS.exist_model_dir
MAX_STEPS = PARAMS.max_steps
LogDevicePlacement = PARAMS.log_device_placement
#LOSSMODE = PARAMS.loss_mode

DATA_TRAIN = PARAMS.dir_train

def restore_model(sess, saver, ExistModelDir, global_step):
    log_info = "Restoring Model From %s..." % ExistModelDir
    print(log_info)
    ckpt = tf.train.get_checkpoint_state(ExistModelDir)
    init_step = 0
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        init_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        sess.run(tf.assign(global_step, init_step))
    else:
        print('No Checkpoint File Found!')
        return
    
    return init_step


    
def train(): 
    global_step = tf.train.get_or_create_global_step()
#    global_step = 1
    # sometimes ending up on GPUs resulting in a slowdown.
    with tf.device('/cpu:0'):
        rawList, segList = unet_input.GetFileNameList(DATA_TRAIN)
        rawImageBatch, segImageBatch = unet_input.GetBatchFromFile_Train(rawList, segList, BATCH_SIZE)
        
    # Build a Graph that computes the predicted HR images from GIBBS RING CLEAR model.
    PredBatch = unet_model.UNet(rawImageBatch)
    
    # Calculate loss.
    TrainLoss = unet_model.loss(segImageBatch, PredBatch)
    
    # Get the training op for optimizing loss
    TrainOp = unet_model.optimize(TrainLoss, LearningRate, global_step)
    
    TrainMeanPSNR = unet_model.evaluation(segImageBatch, PredBatch)
    dice = unet_model.Dice(segImageBatch, PredBatch)
    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())
    
    # Build the summary operation from the last tower summaries.
    summ_op = tf.summary.merge_all()
    
    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU implementations.
    init_step = 0
    config = tf.ConfigProto()
    config.log_device_placement = LogDevicePlacement  #是否打印设备分配日志
    config.allow_soft_placement = AllowSoftPlacement    #如果指定的设备部存在，允许TF自动分配设备。
    with tf.Session(config = config) as sess:
        if TrainFromExist:
            init_step = restore_model(sess, saver, ExistModelDir, global_step)
        else:
            print("Initializing Variables...")
            sess.run(tf.global_variables_initializer())
        
        # queue runners, multi threads and coordinator
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess, coord = coord)
        
        print("Defining Summary Writer...")
        summary_writer = tf.summary.FileWriter(LogDir, sess.graph)
        
        min_loss = float('Inf')
        max_psnr = 0
        max_dice = 0
        try:
            print("Starting To Train...")
            for step in range(init_step, MAX_STEPS):
                # excute training once!
                #start_time = time.time()
                sess.run(TrainOp)
                #duration = time.time() - start_time
            
                if (step + 1) % LogFreq == 0:
                #     examples_per_second = BATCH_SIZE/duration
                #     seconds_per_batch = float(duration)
                    
                    loss_value, PSNR_value, dice_value, batch_raw, image_labels, model_seg = sess.run([TrainLoss, TrainMeanPSNR, dice, rawImageBatch, segImageBatch, PredBatch])
#
                    if min_loss > loss_value:
                        min_loss = loss_value
                    if max_psnr < PSNR_value:
                        max_psnr = PSNR_value
                    if max_dice < dice_value:
                        max_dice = dice_value

                    with open("Records/train_records.txt", "a") as file:
                        format_str = "%d\t%.6f\t%.6f\t%.6f\t%.6f\n"
                        file.write(str(format_str) % (
                        step + 1, loss_value, min_loss, dice_value, max_dice))

                    print("%s ---- step %d:" % (datetime.now(), step + 1))
                    print("\tLOSS = %.6f\tmin_Loss = %.6f" % (loss_value, min_loss))
                    print("\tPSNR = %.4f\tmax_PSNR = %.4f" % (PSNR_value, max_psnr))
                    print("\tDICE = %.2f\tmax_DICS = %.2f" % (dice_value, max_dice))


                if ((step + 1) % 100 == 0) or ((step + 1) == MAX_STEPS):
                    summary_str = sess.run(summ_op)
                    summary_writer.add_summary(summary_str, step + 1)
                    
                if (step == 0) or ((step + 1) % 200 == 0) or ((step + 1) == MAX_STEPS):
                    checkpoint_path = os.path.join(LogDir, 'model.ckpt')
                    print("saving checkpoint into %s-%d" % (checkpoint_path, step + 1))
                    saver.save(sess, checkpoint_path, global_step = step + 1)
                    
        except Exception as e:
            coord.request_stop(e)
            
        finally:
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs = 10)

def main(argv = None):  # pylint: disable = unused - argument
    if not TrainFromExist:
        if tf.gfile.Exists(LogDir):
            tf.gfile.DeleteRecursively(LogDir)
        tf.gfile.MakeDirs(LogDir)
    else:
        if not tf.gfile.Exists(ExistModelDir):
            raise ValueError("Train from existed model, but the target dir does not exist.")
        
        if not tf.gfile.Exists(LogDir):
            tf.gfile.MakeDirs(LogDir)
    train()


if __name__ == '__main__':
    tf.app.run()

    




    
    
    
    
    
    

