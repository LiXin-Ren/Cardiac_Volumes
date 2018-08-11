from __future__ import absolute_import
#import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
import unet_input
import numpy as np
import unet_model
import settings_unet
import time
import os
from sklearn.metrics import mean_squared_error
import cv2
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from scipy import misc

parser = settings_unet.parser
PARAMS = parser.parse_args()

VAL_DIR = PARAMS.dir_validate
# VAL_Raw_DIR = PARAMS.val_Raw_dir
# VAL_Seg_DIR = PARAMS.val_Seg_dir
    
VAL_BATCH_SIZE = 1 # process a single image (NOT patch) one time
VAL_INTERVAL_SECS = 180
Raw_NAME_LIST, Seg_NAME_LIST = unet_input.GetFileNameList(VAL_DIR)
#TOTAL_VALID_IMAGES = len(os.listdir(VAL_DIR)) // 2
INPUT_SIZE = PARAMS.img_input
OUTPUT_SIZE = PARAMS.img_output
rawNameList, segNameList = unet_input.GetFileNameList(VAL_DIR)
TOTAL_VALID_IMAGES = len(rawNameList) 

def EvaluateOnceAndSaveImages(saver, summ_writer, summ_op, seg_image, raw_image, seg_model):
    with tf.Session(config = tf.ConfigProto(log_device_placement = PARAMS.log_device_placement)) as sess:
        #ckpt = tf.train.get_checkpoint_state(train.LogDir)
        ckpt = tf.train.get_checkpoint_state(PARAMS.log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('no checkpoint file found!')
            return
        
        # start the queue runners
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord = coord, daemon = True, start = True))
            
            # calculate the number of iterations for validation set. Because the batch size in this case
            # is 1, we just need to excute TOTAL_VALID_EXAMPLES times loop.
            step = 0
            model_loss = 0
            while not coord.should_stop() and step < TOTAL_VALID_IMAGES:
                print('processing image %d/%d...' % (step + 1, TOTAL_VALID_IMAGES))
                
                image_seg, image_raw, model_seg = sess.run([seg_image, raw_image, seg_model])
                
                image_name = Raw_NAME_LIST[step].split('/')[-1]
                image_path = os.path.join(PARAMS.image_save_dir, image_name)
#                plt.imsave(image_path, model_seg[0, :, :, 0], cmap = "gray") # Note batch_size = 1
#                cv2.imwrite(image_path, model_seg[0, :, :, 0])               
                misc.imsave(image_path, model_seg[0, :, :, 0])
                #pred_seg = tf.reshape(model_seg[0,:,:,0], [184, 184])
                #np.reshape(model_seg[0,:,:,0], (184,184,1))
                #print("*************************",model_seg[0,:,:,0].shape)
                #print('******************', image_seg.shape)
#                tem_loss = unet_model.loss(model_seg[0,:,:,0], image_seg)
                model_seg = np.reshape(model_seg, (184,184))
                image_seg = np.reshape(image_seg, (184,184))
               # tmp_loss = np.square(np.subtract(model_seg, image_seg)).mean()  #MSE
                model_seg_new = np.float32(model_seg > 0.)	#转为1-0
                image_seg_new = np.float32(image_seg > 0.)
                intersection = np.sum(model_seg_new * image_seg_new)
                summ = np.sum(model_seg_new) +np.sum(image_seg_new)                 
                tmp_loss = (2*intersection + 10) /(summ + 10)
                with open("Records/validate_records.txt", "a") as file:
                    #format_str = "%d\t%.6f\t%.6f\t%.6f\t%.6f\n"
#                    file.write(str(format_str) % (
#                    step + 1, loss_value, min_loss, dice_value, max_dice))
                    file.write(str("%d\t%.4f\n") %(step+1, tmp_loss))

#                tmp_loss = ((model_seg - image_seg) ** 2).mean() # mean_squared_error(model_seg, image_seg)
                print(" ---- %s:\n\tmodel_loss = %.4f" % (image_name, tmp_loss))
           #     print(" \tgibbs_psnr = %.4f\tgibbs_ssim = %.4f" % (before_psnr, before_ssim))
                model_loss += tmp_loss
               
                step += 1
            
            model_loss = model_loss / TOTAL_VALID_IMAGES
           
            print("%s: -- Validation Set:" % datetime.now())
            print("\tmean_model_loss = %.4f\t" % (model_loss))
           
            # writer relative summary into val_log_dir
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summ_op))
            summary.value.add(tag='Average model MSE over validation set', simple_value = model_loss)
          
            summ_writer.add_summary(summary, global_step)
        except Exception as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs = 10)
            

def EvaluateOnce(saver, summ_writer, summ_op, clear_image, gibbs_image, clear_model):
    with tf.Session(config = tf.ConfigProto(log_device_placement = PARAMS.log_device_placement)) as sess: 
        # Synchronous assessment: it should use train logs train.LogDir here!
        ckpt = tf.train.get_checkpoint_state(PARAMS.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('no checkpoint file found!')
            return
        
        # start the queue runners
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord = coord, daemon = True, start = True))
            
            # calculate the number of iterations for validation set
            model_psnr = 0.0
            model_ssim = 0.0
            gibbs_psnr = 0.0
            gibbs_ssim = 0.0
            step = 0
            while not coord.should_stop() and step < TOTAL_VALID_IMAGES:
                print('processing image %d/%d...' % (step + 1, TOTAL_VALID_IMAGES))
                
                image_CLEAR, image_GIBBS, model_CLEAR = sess.run([clear_image, gibbs_image, clear_model])
                tmp_psnr, tmp_ssim = cal_psnr_and_ssim(image_CLEAR, model_CLEAR)
                before_psnr, before_ssim = cal_psnr_and_ssim(image_CLEAR, image_GIBBS)
                model_psnr += tmp_psnr
                model_ssim += tmp_ssim
                gibbs_psnr += before_psnr
                gibbs_ssim += before_ssim
                
                step += 1
            
            # calculate the average PSNR over the whole validation set
            model_psnr = model_psnr / TOTAL_VALID_IMAGES
            model_ssim = model_ssim / TOTAL_VALID_IMAGES
            gibbs_psnr = gibbs_psnr / TOTAL_VALID_IMAGES
            gibbs_ssim = gibbs_ssim / TOTAL_VALID_IMAGES
            print("%s: -- Validation Set:" % datetime.now())
            print("\tmodel_psnr = %.4fdB\tmodel_ssim = %.4fdB" % (model_psnr, model_ssim))
            print("\tgibbs_psnr = %.4fdB\tgibbs_ssim = %.4fdB" % (gibbs_psnr, gibbs_ssim))
            
            # writer relative summary into val_log_dir
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summ_op))
            summary.value.add(tag='Average model PSNR over validation set', simple_value = model_psnr)
            summary.value.add(tag='Average model PSNR over validation set', simple_value = model_ssim)
            summary.value.add(tag='Average gibbs PSNR over validation set', simple_value = gibbs_psnr)
            summary.value.add(tag='Average gibbs PSNR over validation set', simple_value = gibbs_ssim)
            summ_writer.add_summary(summary, global_step)
        except Exception as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs = 10)


def Evaluate():
    with tf.Graph().as_default() as g:
#        rawNameList, segNameList = unet_input.GetFileNameList(VAL_DIR)
#       TOTAL_VALID_IMAGES = len(rawNameList)
        raw_image, seg_image = unet_input.GetBatchFromFile_Valid(rawNameList, segNameList, VAL_BATCH_SIZE)
                
        # Build computational graph
        seg_model = unet_model.UNet(raw_image)
        #raw_image = tf.image.resize_bicubic(raw_image, [OUTPUT_SIZE, OUTPUT_SIZE])
        saver = tf.train.Saver()
        
        summ_op = tf.summary.merge_all()
        summ_writer = tf.summary.FileWriter(PARAMS.val_log_dir, g)
        while True:
            if PARAMS.eval_once:
                EvaluateOnceAndSaveImages(saver, summ_writer, summ_op, seg_image, raw_image, seg_model)
                break
            else:
                EvaluateOnce(saver, summ_writer, summ_op, seg_image, raw_image, seg_model)
                time.sleep(VAL_INTERVAL_SECS)
        

def main(argv = None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(PARAMS.val_log_dir):
        tf.gfile.DeleteRecursively(PARAMS.val_log_dir)
    tf.gfile.MakeDirs(PARAMS.val_log_dir)
    Evaluate()


if __name__ == '__main__':
    tf.app.run()   
