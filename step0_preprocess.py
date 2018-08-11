__author__ = 'Lisa'
import helpers
import settings
import csv
import pandas
from helpers_dicom import *
import scipy
import scipy.misc
import cv2
from PIL import Image

DATADIR = settings.TRAIN_DIR

def create_csv_data(data_dir):
    print("Creating csv file from dicom data")
    row_no = 0
    with open("dicom_data.csv", "w", newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["patient_id", "slice_no", "frame_no",
                             "spacing","slice_thickness", "slice_location", "time"])

        for dicom_data in helpers.enumerate_sax_files(data_dir):
            row_no += 1
            if row_no % 1000 == 0:
                print(row_no)

            csv_writer.writerow([
                str(dicom_data.patient_id),                         #病人编号
                str(dicom_data.series_number),                      #层号，与文件夹数字一致，eg：sax_5中的文件seriesNumber都是5
                str(dicom_data.instance_number),                    #在同一层中的不同时间下的编号。
                str(dicom_data.spacing[0]),                         #像素的物理间距，即实际大小
                str(dicom_data.slice_thickness),                    #层厚
                str(dicom_data.slice_location),                     #当前层的位置，单位mm, 同一层一致
                str(dicom_data.get_value("InstanceCreationTime")),  #当前图片的时间
            ])


def up_down(current_value, previous_value):
    # previous_value = previous_value.fillna(-99999)
    delta = current_value - previous_value
    delta = delta.fillna(0)
    updown = pandas.Series(delta.apply(lambda x: 0 if x == 0 else 1 if x > 0 else -1))
    return updown


def slice_delta(current_value, next_value):
    # previous_value = previous_value.fillna(-99999)
    delta = current_value - next_value
    delta = delta.fillna(999)
    return delta


def count_small_deltas(current_value):
    # previous_value = previous_value.fillna(-99999)
    res = len(current_value[abs(current_value) < 2])
    return res



def enrich_dicom_csvdata():
    print("Enriching dicom csv data with extra columns and stats")
    dicom_data = pandas.read_csv(settings.BASE_DIR + "dicom_data.csv", sep=",", engine='python')
    dicom_data["patient_id_frame"] = dicom_data["patient_id"].map(str) + "_" + dicom_data["frame_no"].map(str)
    dicom_data = dicom_data.sort_values(["patient_id", "frame_no", "slice_location"], ascending=[1, 1, 1])

    # aggrageted updown information < 0 means slice location increased from apex to base and > 0 from base to apex,
    # we want everything from base to apex..
    patient_grouped = dicom_data.groupby("patient_id_frame")
    dicom_data['up_down'] = patient_grouped['time'].apply(lambda x: up_down(x, x.shift(1)))
    dicom_data['up_down_agg'] = patient_grouped["up_down"].transform(lambda x: sum(x))
    dicom_data['slice_location_sort'] = dicom_data['slice_location'] * dicom_data['up_down_agg']
    dicom_data = dicom_data.sort_values(["patient_id", "frame_no", "slice_location_sort", "slice_location"])

    # now compute the deltas between slices
    patient_grouped = dicom_data.groupby("patient_id_frame")
    dicom_data['slice_location_delta'] = patient_grouped['slice_location'].apply(lambda x: slice_delta(x, x.shift(-1)))
   # dicom_data['small_slice_count'] = patient_grouped['slice_location_delta'].transform(lambda x: count_small_deltas(x))
   # dicom_data["slice_count"] = patient_grouped["up_down"].transform("count")
   # dicom_data["normal_slice_count"] = dicom_data["slice_count"] - dicom_data['small_slice_count']

    # delete all slices with delta '0'
    #dicom_data = dicom_data[dicom_data["slice_location_delta"] != 0]

    # again determine updown for some special cases (341)
    patient_grouped = dicom_data.groupby("patient_id_frame")
    dicom_data['up_down'] = patient_grouped['time'].apply(lambda x: up_down(x, x.shift(1)))
    dicom_data['up_down_agg'] = patient_grouped["up_down"].transform(lambda x: sum(x))

    dicom_data.to_csv(settings.BASE_DIR + "dicom_data_enriched.csv", sep=",")

    dicom_data = dicom_data[dicom_data["frame_no"] == 1]
    dicom_data.to_csv(settings.BASE_DIR + "dicom_data_enriched_frame1.csv", sep=",")


def enrich_traindata():
    print("Enriching train data with extra columns and stats")
    train_data = pandas.read_csv(settings.BASE_DIR + "train_validate.csv", sep=",")
    dicom_data = pandas.read_csv(settings.BASE_DIR + "dicom_data_enriched_frame1.csv", sep=",")
    patient_grouped = dicom_data.groupby("patient_id")

    enriched_traindata = patient_grouped.first().reset_index()
    enriched_traindata = enriched_traindata[["patient_id", "spacing", "slice_thickness",  "up_down_agg"]]
    enriched_traindata = pandas.merge(left=enriched_traindata, right=train_data, how='left', left_on='patient_id', right_on='Id')

    enriched_traindata["pred_dia"] = 0
    enriched_traindata["error_dia"] = 0
    enriched_traindata["abserr_dia"] = 0
    enriched_traindata["pred_sys"] = 0
    enriched_traindata["error_sys"] = 0
    enriched_traindata["abserr_sys"] = 0

    enriched_traindata.to_csv(settings.BASE_DIR + "train_enriched.csv", sep=",")


def get_patient_id(dir):
    parts = dir.split('/')
    res = parts[len(parts) - 3]
    return res


def get_slice_type(dir_name):
    parts = dir_name.split('/')
    res = parts[len(parts) - 1]
    return res


###可用tf.image.resize_with_crop_or_pad()_
#中心裁剪
def get_square_crop(img, crop_size=256):
    res = img
    height, width = res.shape

    #填充
    if height < crop_size:
        diff = crop_size - height
        extend_top = diff // 2
        extend_bottom = diff - extend_top
        res = cv2.copyMakeBorder(res, extend_top, extend_bottom, 0, 0, borderType=cv2.BORDER_CONSTANT, value=0)
        #making borders for image
        height = crop_size

    if width < crop_size:
        diff = crop_size - width
        extend_top = diff // 2
        extend_bottom = diff - extend_top
        res = cv2.copyMakeBorder(res, 0, 0, extend_top, extend_bottom, borderType=cv2.BORDER_CONSTANT, value=0)
        width = crop_size

    crop_y_start = (height - crop_size) // 2
    crop_x_start = (width - crop_size) // 2
    res = res[crop_y_start:(crop_y_start + crop_size), crop_x_start:(crop_x_start + crop_size)]
    return res


def convert_sax_images(dicom_dir, rescale=True, crop_size=256):
    """
    :param dicom_dir: 文件路径
    :param rescale:
    :param base_size:
    :param crop_size:
    :return:
    """
    """读取dicom文件，并将其转换为png格式，裁剪，翻转等，然后将图像存储起来"""
    target_dir = settings.BASE_PREPROCESSEDIMAGES_DIR

    file_count = 0
    for dicom_data in helpers.enumerate_sax_files(dicom_dir):
        file_count += 1

        if dicom_data.in_plane_encoding_direction not in ["ROW", "COL"]:
            print("Error: plane_encoding_direction", dicom_data.file_name)
            raise Exception("ROW,COL")

        if dicom_data.spacing[0] != dicom_data.spacing[1]:
            print("Error: spacing")
            raise Exception("Data spacings not equal")

        if file_count % 100 == 0:
            print(str(dicom_data.patient_id) + "\t" + str(dicom_data.series_number))
                  

        location_id = int(dicom_data.slice_location) + 1000   #负数变正数
        location_id_str = str(location_id).rjust(4, '0')

        #rjust 字符串的长度统一
        #命名规则
        img_path = target_dir + str(dicom_data.patient_id).rjust(4, '0') + '_'\
                   + str(dicom_data.series_number).rjust(2, '0') +"_" \
                   +str(dicom_data.instance_number).rjust(3,'0') +  ".png"

        #img = Image.fromarray(dicom_data.pixel_array)       #数组转图像
        scipy.misc.imsave(img_path, dicom_data.pixel_array) #将数组存为图像

        img = cv2.imread(img_path, 0)
        if dicom_data.in_plane_encoding_direction == "COL":
           # rotate counter clockwise when image is column oriented..
            img = cv2.transpose(img)    #图像转置，将x，y互换
            img = cv2.flip(img, 0)      #纵向翻转图像

        if rescale:         #将图像从像素大小转为物理大小
            scale = dicom_data.spacing[0]
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

        sq_img = get_square_crop(img, crop_size=crop_size)
        clahe = cv2.createCLAHE(tileGridSize=(1, 1))    #直方图均衡化，以小块为单位
        cl_img = clahe.apply(sq_img)
        cv2.imwrite(img_path, cl_img)


if __name__ == "__main__":
    #convert_sax_images(DATADIR, rescale=True, crop_size=256)
    #create_csv_data(DATADIR)
    enrich_dicom_csvdata()
    #enrich_traindata()
    print("data convert Done")
