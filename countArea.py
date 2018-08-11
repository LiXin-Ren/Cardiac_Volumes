import cv2
import csv
import numpy as np
import os
from skimage.measure import label

seg_path = "seged_images/"

csv_name = "dicom_data.csv"

picture_path = ''

def countPixel(path):
    if not os.path.exists(path):
        print(path + 'is not exists')
        return 'area'
    img = cv2.imread(path, 0)
    img1 = img > 0
   
    connetArea, numConnective = label(img1, neighbors = 8, return_num = True)
    area = 0
    for i in range(numConnective):
        area = max(area, np.sum(img==(i+1)))
    return area

with open('dicom_data.csv') as csvfile:
    rows = list(csv.reader(csvfile))
    with open('dicom_data2.csv', 'w', newline='') as csvfileW:
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
            location_id = int(float(rows[i][9])) + 1000  # 负数变正数
            location_id_str = str(location_id).rjust(4, '0')

            target_path = seg_path + rows[i][0].rjust(4, '0') + '_' \
                   + location_id_str +"_" \
                   + rows[i][16] + ".png"

            area = countPixel(target_path)
            
            rows[i].append(area)
            myWrite.writerow(rows[i])
