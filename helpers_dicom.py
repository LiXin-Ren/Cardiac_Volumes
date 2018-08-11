__author__ = 'Julian'
import numpy as np
import pydicom


class DicomWrapper:
    def __init__(self, file_dir, file_name):
        self.raw_file = pydicom.dcmread(file_dir + file_name)
        self.file_name = file_name

    def get_value(self, name):
        res = self.raw_file.data_element(name).value
        return res

    @property
    def patient_id(self):   #（0010， 0020）
        return self.get_value("PatientID")

    @property
    def columns(self):  #（0028， 0011）
        res = self.get_value("Columns")
        return res

    @property
    def rows(self):     #（0028， 0010）
        res = self.get_value("Rows")
        return res

    #像素间距：像素中心的物理间距
    @property
    def spacing(self):
        res = self.get_value("PixelSpacing")    #（0028， 0030）
        return res

    @property
    def slice_location(self):
        return self.get_value("SliceLocation")  #实际相对位置（0020， 1041）

    @property
    def image_position(self):
        return self.get_value("ImagePositionPatient")  # 图像左上角第一个像素的空间坐标（0020， 0032）

    @property
    def image_orientation_patient(self):    #（0020， 0037）图像方向
        return self.get_value("ImageOrientationPatient")

    @property
    def slice_thickness(self):  #层厚（0018， 0050）
        return self.get_value("SliceThickness")

    @property
    def series_number(self):
        return self.get_value("SeriesNumber")   #（0020， 0011）

    @property
    def series_time(self):
        return self.get_value("SeriesTime") #（0008， 0031）

    @property
    def series_description(self):
        return self.get_value("SeriesDescription")  #eg：sax（0008， 103E)

    @property
    def flip_angle(self):
        return self.get_value("FlipAngle")  #偏转角度(0018, 1314)

    @property
    def instance_number(self):
        return self.get_value("InstanceNumber") #图像号码（0020， 0013）

    @property
    def in_plane_encoding_direction(self):  #编码方向（0018， 1312）row or column
        return self.get_value("InPlanePhaseEncodingDirection")

    @property
    def window_center(self):   #窗位（0028， 1050）
        window_center = self.get_value("WindowCenter")
        return window_center

    @property
    def window_width(self):    #窗宽（0028， 1051）
        window_width = self.get_value("WindowWidth")
        return window_width

    @property
    def patient_position(self):  # 病人躺的方位（0018， 5100）
        patient_position = self.get_value("PatientPosition")
        return patient_position


    def get_location(self):
        image_center2d = self.spacing * (np.array([self.columns, self.rows]) - np.ones(2)) / 2.
        image_center3d = np.dot(image_center2d, np.reshape(self.image_orientation_patient, (2, 3)))
        center = self.image_position + image_center3d
        direction = np.argmax(np.abs(np.cross(self.image_orientation_patient[:3], self.image_orientation_patient[3:])))
        #np.cross叉乘（点积）
        res = np.round(center[direction], 2)
        return center

    # @property
    # def pixel_array(self):
    #     img = self.raw_file.pixel_array.astype(float) / numpy.max(self.raw_file.pixel_array)
    #     return img

    @property
    def pixel_array(self):
        img = self.raw_file.pixel_array.astype(int)
        return img

    def get_csv(self):
        res = [self.series_number, self.instance_number, self.flip_angle, self.series_description, self.series_time]
        return res

    def dir(self):
        self.raw_file.dir()


    # @property
    # def sequence_name(self):
    #     return self.get_value("SequenceName")

    # @property
    # def create_time(self):    #（0008， 0013）
    #     return str(int(round(float(self.get_value("InstanceCreationTime")))) / 10).rjust(5, '0')


    # @property
    # def image_position_patient(self):   #（0020， 0032）
    #     return self.get_value("ImagePositionPatient")
#
# ['AcquisitionMatrix',
#  'AcquisitionNumber',
#  'AcquisitionTime',
#  'AngioFlag',
#  'BitsAllocated',
#  'BitsStored',
#  'BodyPartExamined',
#  'CardiacNumberOfImages',
#  'Columns',
#  'CommentsOnThePerformedProcedureStep',
#  'EchoNumbers',
#  'EchoTime',
#  'EchoTrainLength',
#  'FlipAngle',
#  'HighBit',
#  'ImageOrientationPatient',
#  'ImagePositionPatient',
#  'ImageType',
#  'ImagedNucleus',
#  'ImagingFrequency',
#  'InPlanePhaseEncodingDirection',
#  'InstanceCreationTime',
#  'InstanceNumber',
#  'LargestImagePixelValue',
#  'MRAcquisitionType',
#  'MagneticFieldStrength',
#  'Manufacturer',
#  'ManufacturerModelName',
#  'Modality',
#  'NominalInterval',
#  'NumberOfAverages',
#  'NumberOfPhaseEncodingSteps',
#  'PatientAddress',
#  'PatientAge',
#  'PatientBirthDate',
#  'PatientID',
#  'PatientName',
#  'PatientPosition',
#  'PatientSex',
#  'PatientTelephoneNumbers',
#  'PercentPhaseFieldOfView',
#  'PercentSampling',
#  'PerformedProcedureStepID',
#  'PerformedProcedureStepStartTime',
#  'PhotometricInterpretation',
#  'PixelBandwidth',
#  'PixelData',
#  'PixelRepresentation',
#  'PixelSpacing',
#  'PositionReferenceIndicator',
#  'RefdImageSequence',
#  'ReferencedImageSequence',
#  'RepetitionTime',
#  'Rows',
#  'SAR',
#  'SOPClassUID',
#  'SOPInstanceUID',
#  'SamplesPerPixel',
#  'ScanOptions',
#  'ScanningSequence',
#  'SequenceName',
#  'SequenceVariant',
#  'SeriesDescription',
#  'SeriesNumber',
#  'SeriesTime',
#  'SliceLocation',
#  'SliceThickness',
#  'SmallestImagePixelValue',
#  'SoftwareVersions',
#  'SpecificCharacterSet',
#  'StudyTime',
#  'TransmitCoilName',
#  'TriggerTime',
#  'VariableFlipAngleFlag',
#  'WindowCenter',
#  'WindowCenterWidthExplanation',
#  'WindowWidth',
#  'dBdt']

