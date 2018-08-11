import pandas as pd
import numpy as np
from scipy.stats import norm
import cv2
import csv
import os
#from skimage.measure import label

seg_path = "seged_images/"

raw_csv_name = "dicom_data.csv"

picture_path = ''

def countPixel(path):
    if not os.path.exists(path):
        print(path + 'is not exists')
        return 'area'
    img = cv2.imread(path, 0)
   # img1 = img > 0
    area = np.sum(img>0)   
   # connetArea, numConnective = label(img1, neighbors = 8, return_num = True)
   # area = 0
   # for i in range(numConnective):
   #     area = max(area, np.sum(img==(i+1)))
    return area


def countArea():
    with open('dicom_data.csv') as csvfile:
        rows = list(csv.reader(csvfile))
        with open('dicom_data1.csv', 'w', newline='') as csvfileW:
            myWrite = csv.writer(csvfileW, delimiter=',')
        # for row in rows:
        #     location_id = int(float(row[9])) + 1000  # 负数变正数
        #     location_id_str = str(location_id).rjust(4, '0')
        #     target_path = seg_path + row[0].rjust(4, '0') + '_' \
        #            + location_id_str +"_" \
        #            + row[15] + ".png"
        #
        #     area = countPixel(target_path)
        #     row.append(area)
        #     myWrite.writerow(row)
            row = rows[0] + ['area']
            myWrite.writerow(row)

            for i in range(1, len(rows)):
                if i % 1000 == 0:
                    print(i)
#           frame_no = int(float(rows[i][2])).rjust(3, '0') 
#            slice_no = int(rows[i][1]).rjust(2, '0')

                target_path = seg_path + rows[i][0].rjust(4, '0') + '_' \
                   + rows[i][1].rjust(2,'0') +"_" \
                   + rows[i][2].rjust(3,'0') + ".png"

                area = countPixel(target_path)
            
                rows[i].append(area)
                myWrite.writerow(rows[i])


def compute_height(curr, next):
    delta = curr - next
    delta = delta.fillna(0)  # 0填充nan
    #updown = pd.Series(delta.apply(lambda x: 0 if x == 0 else 1 if x > 0 else -1))
    return delta

def enrich_dicom_csvdata():
#获取求volumn的一些参数，如高度、层间距等
    print("Enriching dicom csv data with extra columns and stats")
    #dicom_data = pandas.read_csv(settings.BASE_DIR + "dicom_data.csv", sep=",", engine='python')
    dicom_data = pd.read_csv("dicom_data1.csv", sep=",", engine='python')
    newDf = pd.DataFrame(columns=['slice_max'])
    dicom_data["patient_id_slice"] = dicom_data["patient_id"].map(str) + "_" + dicom_data["slice_no"].map(str)
    dicom_data = dicom_data.sort_values(["patient_id", "slice_location", "slice_no", "frame_no"], ascending=[1, 1, 1, 1])

    # aggrageted updown information < 0 means slice location increased from apex to base and > 0 from base to apex,
    # we want everything from base to apex..
    patient_grouped = dicom_data.groupby("patient_id_slice")
    #dicom_data['up_down'] = patient_grouped['time'].apply(lambda x: up_down(x, x.shift(1))) #与下一层相减
    #dicom_data['up_down_agg'] = patient_grouped["up_down"].transform(lambda x: sum(x))
    #dicom_data['slice_location_sort'] = dicom_data['slice_location'] * dicom_data['up_down_agg']
    #dicom_data = dicom_data.sort_values(["patient_id", "frame_no", "slice_location_sort", "slice_location"])

    newDf['slice_max'] = patient_grouped['area'].apply(lambda x: max(x))
    newDf['slice_min'] = patient_grouped['area'].apply(lambda x: min(x))
    newDf['slice_thickness'] = patient_grouped['slice_thickness'].mean()
    newDf['slice_location'] = patient_grouped['slice_location'].mean()
    newDf.to_csv("dicom_data_slice(apply1).csv", sep=",")

    data = pd.read_csv('dicom_data_slice(apply1).csv')

    data['patient_id'] = data["patient_id_slice"].apply(lambda x: int(x.split('_')[0]))
    data['slice'] = data["patient_id_slice"].apply(lambda x: x.split('_')[1])
    data = data.sort_values(["patient_id", "slice_location"], ascending=[1, 1])
    data.to_csv("dicom_data_slice(apply1).csv", sep=",")

    data = pd.read_csv('dicom_data_slice(apply1).csv')
    grouped_data = data.groupby('patient_id')

    data['height'] = grouped_data['slice_location'].apply(lambda x: compute_height(x, x.shift(1)))

    # now compute the deltas between slices
    # patient_grouped = dicom_data.groupby("patient_id_slice")
    # dicom_data['slice_location_delta'] = patient_grouped['slice_location'].apply(lambda x: slice_delta(x, x.shift(-1)))
    # dicom_data['small_slice_count'] = patient_grouped['slice_location_delta'].transform(lambda x: count_small_deltas(x))
    # dicom_data["slice_count"] = patient_grouped["up_down"].transform("count")
    # dicom_data["normal_slice_count"] = dicom_data["slice_count"] - dicom_data['small_slice_count']
    #
    # # delete all slices with delta '0'
    # dicom_data = dicom_data[dicom_data["slice_location_delta"] != 0]
    #
    # # again determine updown for some special cases (341)
    # patient_grouped = dicom_data.groupby("patient_id_slice")
    # dicom_data['up_down'] = patient_grouped['time'].apply(lambda x: up_down(x, x.shift(1)))
    # dicom_data['up_down_agg'] = patient_grouped["up_down"].transform(lambda x: sum(x))

    data.to_csv("dicom_data_slice(apply1).csv", sep=",")
    #
    # dicom_data = dicom_data[dicom_data["frame_no"] == 1]
    # dicom_data.to_csv(settings.BASE_DIR + "dicom_data_enriched_frame1.csv", sep=",")
    

def compute_volumns():
    data = pd.read_csv('dicom_data_slice(apply1).csv')

    data['slice_max_volumns'] = data['height'] * data['slice_max']
    data['slice_min_volumns'] = data['height'] * data['slice_min']

    data.to_csv("dicom_data_slice(slice_volumns).csv", sep=",")


    data = pd.read_csv('dicom_data_slice(slice_volumns).csv')

    patien_data = data.groupby("patient_id")
    newDF = pd.DataFrame(columns=['max_volumns', 'min_volumns'])
    newDF['pred_Diatole'] = round(patien_data['slice_max_volumns'].sum() / 1000, 1)
    newDF['pred_Systole'] = round(patien_data['slice_min_volumns'].sum() / 1000, 1)
    newDF.to_csv("dicom_data_slice(volumns).csv", sep=",")


def crps(true, pred):
    """
    Calculation of CRPS.
    :param true: true values (labels)
    :param pred: predicted values
    """
    return np.sum(np.square(true - pred)) / true.size


def real_to_cdf(y, sigma=1e-10):
    """
    Utility function for creating CDF from real number and sigma (uncertainty measure).
    :param y: array of real values
    :param sigma: uncertainty measure. The higher sigma, the more imprecise the prediction is, and vice versa.
    Default value for sigma is 1e-10 to produce step function if needed.
    """
    cdf = np.zeros((y.shape[0], 600))
    for i in range(y.shape[0]):
        cdf[i] = norm.cdf(np.linspace(0, 599, 600), y[i], sigma)
    return cdf




if __name__ == "__main__":
   # countArea() 
   # enrich_dicom_csvdata()
   # compute_volumns()


    data = pd.read_csv('dicom_data_slice(volumns).csv')
    pred_Diastole = real_to_cdf(data['pred_Diastole'])
    real_Diatole = real_to_cdf(data['Diastole'])

    pred_Systole = real_to_cdf(data['pred_Systole'])
    real_Systole = real_to_cdf(data['Systole'])

    crpsSys = crps(pred_Systole, real_Systole)
    crpsDia = crps(pred_Diastole, real_Diatole)
    print('Dia: ', crpsDia)
    print('Sys: ', crpsSys)
    print("data convert Done")

