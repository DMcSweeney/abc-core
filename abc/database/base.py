"""
Database parent class containing util. methods for cleanliness
"""
from multiprocessing.sharedctypes import Value
import os
import ast
import math
from re import L
from xmlrpc.client import MultiCall
import SimpleITK as sitk
import numpy as np
import pandas as pd
from colorama import Fore, Style, Back
import uuid as uid
import matplotlib.pyplot as plt
from tqdm import tqdm
import skimage
import scipy
import time
import logging

from sqlalchemy import insert, update
from ..mixin import dotdict
from datetime import datetime

logger = logging.getLogger(__name__)

class DataBase_Helper():
    def __init__(self):
        ...

    #~ === I/O ====
    def parse_directory(self, filename):
        #~ When input is set of directory, use this to find path to image
        #~ Gets all images for a patient 
        #* Extract info. from intermediate directories if any.
        path = os.path.join(self.input_dir, filename) #! Parent -> Patient_ID
        im_paths = []
        im_modalities = []
        im_acquisition_dates = []

        for root, dirs, files in os.walk(path):
            if files:
                #* Check all same file ending
                if all([x.endswith(self.file_extension) for x in files]):
                    """
                    #! Assumes structure: Parent -> Patient_ID -> Modality -> Acquisition_date -> Image
                    #! or: Parent -> Patient_ID -> Modality -> Image
                    #! or: Parent -> Patient_ID -> Image
                    """
                
                    #~ Get path to file
                    if self.file_extension in ['.nii', '.npy', '.nii.gz']:
                        #! Assumes image per directory
                        out_path = os.path.join(root, files[0])
                    elif self.file_extension == '.dcm':
                        out_path = root
                    else:
                        raise ValueError(f'{self.file_extension} not supported.')
                    
                    #~ Get info from directory
                    extra_path = root.lstrip(path)
                    splits = extra_path.split('/')
                    if extra_path != '':
                        if len(splits) == 1: 
                            #! Parent -> Patient_ID -> Modality -> Image
                            modality = splits[0]
                            im_paths.append(out_path)
                            im_modalities.append(modality)
                            im_acquisition_dates.append(None)
                        elif len(splits) == 2: 
                            #! Parent -> Patient_ID -> Modality -> Acquisition_date -> Image
                            modality, acq_date = splits
                            #@ Convert to %Y%m%d
                            try:
                                acq_date = datetime.strptime(acq_date, '%Y%m%d').date()
                            except ValueError:
                                acq_date = datetime.strptime(acq_date, '%d.%m.%Y').strftime('%Y%m%d').date()

                            im_paths.append(out_path)
                            im_modalities.append(modality)
                            im_acquisition_dates.append(acq_date)
                        elif len(splits) == 0:
                            im_paths.append(out_path)
                            im_modalities.append(self.modality)
                            im_acquisition_dates.append(None)
                        else:
                            raise ValueError('Input directory structure not supported yet...')
                    else:
                        #!Parent->Patient_ID -> Image
                        im_paths.append(out_path)
                        im_modalities.append(self.modality)
                        im_acquisition_dates.append(None)

        output = {
            'path': im_paths,
            'modality': im_modalities,
            'acquisition_date': im_acquisition_dates
        }
        return output

    def parse_dicom_directory(self, filename):
        #~ From messy dicom directory, extract useful info from DCM header
        path = os.path.join(self.input_dir, filename) #! Parent -> filename=Patient_ID
        im_paths = []
        im_modalities = []
        im_acquisition_dates = []
        series_uuids = []
        study_uuids = []
        pids = []
        for root, dirs, files in os.walk(path, topdown=True):
            if files and all([x.endswith(self.file_extension) for x in files]):
                #* Check all .dcm
                # Read the first .dcm file
                
                path = os.path.join(root, files[0])

                reader = sitk.ImageFileReader()
                reader.LoadPrivateTagsOn()
                reader.SetFileName(path)
                reader.ReadImageInformation()
                metadata = {}
                for k in reader.GetMetaDataKeys():
                    v = reader.GetMetaData(k)
                    metadata[k] = v
                ## Collect metadata
                series_uuid = metadata['0020|000e']
                study_uuid = metadata['0020|000d']
                modality = metadata['0008|0060']
                if modality not in self.accepted_modalities: 
                    continue
                try:
                    acq_date = metadata['0008|0022']
                except KeyError:
                    acq_date = None

                if acq_date is not None:
                    try:
                        acq_date = datetime.strptime(acq_date, '%Y%m%d').date()
                    except ValueError:
                        try:
                            acq_date = datetime.strptime(acq_date, '%d.%m.%Y').strftime('%Y%m%d').date()
                        except ValueError:
                            acq_date = None
                
                pids.append(filename)
                im_paths.append(root)
                im_modalities.append(modality)
                im_acquisition_dates.append(acq_date)
                series_uuids.append(series_uuid.strip())
                study_uuids.append(study_uuid.strip())
        output = {
            'path': im_paths,
            'modality': im_modalities,
            'acquisition_date': im_acquisition_dates,
            'series_uuid': series_uuids,
            'study_uuid': study_uuids,
            'pid': pids
        }
        return output

    def get_file_extension_from_directory(self, path):
        #~ When input is per-patient directory, check file extension 
        for root, dirs, files in os.walk(path):
            if files:
                if files[0].endswith('.dcm'):
                    self.file_extension = '.dcm'
                    self.loader_function = self.load_dcm
                    return
                elif files[0].endswith('.nii'):
                    self.file_extension = '.nii'
                    self.loader_function = self.load_nifty
                    return
                elif files[0].endswith('.npy'):
                    self.file_extension = '.npy'
                    self.loader_function = self.load_numpy
                    return
                else:
                    self.file_extension = self.files[0].split('.')[-1].strip()
                    raise ValueError(f'{self.file_extension} not supported.')
    
    def add_pid_to_db(self, **kwargs):
        # Check all columns are equal
        num_ims = set([len(val) for val in kwargs.values()])
        # Checks all inputs are the same length i.e. the size of length set is 1
        assert len(num_ims)==1, print("Args are mismatched, can't update db. Size: ", len(num_ims))
        for i in range(list(num_ims)[0]):
            data = {}
            for key, val in kwargs.items():
                data[key] = val[i]
            self.update_tables(**data)

    #~ === DATABASE UTILITIES =====
    def update_tables(self, **in_kwargs):
        #@ Insert one image into tables
        #@ Kwargs should be at least pid, path, modality, acquisition date
        #* Get original image attributes
        if self.verbose:
            logger.info(f"Reading from: { in_kwargs['path'] }")
        Image = self.loader_function(in_kwargs['path']) #* original image
        Image, orient = self.reorient(Image)
        Image_attr = self.get_image_attributes(Image)
        #*=== Update Image Table ===
        #* Check if entry exists, if it does re-use uuid
        kwargs = {
            'pid': in_kwargs['pid'],
            'path_to_im': in_kwargs['path'],
            'file_extension': self.file_extension,
            'modality': in_kwargs['modality'] if 'modality' in in_kwargs else self.modality,
            'acquisition_date': in_kwargs['acquisition_date'] if 'acquisition_date' in in_kwargs else None,
            'series_uuid': in_kwargs['series_uuid'] if 'series_uuid' in in_kwargs else None,
            'study_uuid': in_kwargs['study_uuid'] if 'study_uuid' in in_kwargs else None,
            'flip_axis': orient.GetFlipAxes()[-1], # Check if z axis was flipped
            **Image_attr
            }
        unique_id = self.update_table(self.tables.images, unique_id = None, **kwargs)
        
        # #~=== Update Interim Table ===
        # new_size = (*self.crop_size, ast.literal_eval(Image_attr['size'])[-1])
        # #* Get attributes for inverse resample
        # new_spacing = [pix*(x/y) for pix, x, y in zip(Image.GetSpacing(), \
        #     new_size, self.output_shape)]

        # #* Record metadata to interim table
        # kwargs = {
        #     'pid': in_kwargs['pid'],
        #     'modality': in_kwargs['modality'] if 'modality' in in_kwargs else self.modality,
        #     'spacing': tuple(new_spacing),
        #     'direction': Image.GetDirection(),
        #     'size': self.output_shape,
        #     'origin': Image.GetOrigin()
        #     }
        # kwargs = {k: str(v) for k, v in kwargs.items()}
        # _ = self.update_table(self.tables.interim,  unique_id = unique_id, **kwargs)

        #~=== Update Outputs Table === 
        #* Add info to outputs table
        kwargs = {
            'pid': in_kwargs['pid'],
            'modality': in_kwargs['modality'] if 'modality' in in_kwargs else self.modality,
            'acquisition_date': in_kwargs['acquisition_date'] if 'acquisition_date' in in_kwargs else None,
            'series_uuid': in_kwargs['series_uuid'] if 'series_uuid' in in_kwargs else None,
            'study_uuid': in_kwargs['study_uuid'] if 'study_uuid' in in_kwargs else None
        }
        kwargs = {k: str(v) for k, v in kwargs.items()}
        _ = self.update_table(self.tables.outputs,  unique_id = unique_id, **kwargs)

    def update_table(self, table, unique_id = None, **kwargs):
        #~ Check entries: update if exists & insert if not 
        with self.session as sess:
            stmt = sess.query(table).filter_by(**kwargs).first()
        if stmt:
            # If match found, re-use uuid & update
            if self.verbose:
                logger.info(f"~ Entry for {kwargs['pid']} in {table} exists, updating ~")
            uuid = stmt.uuid
            with self.session as sess:
                sess.query(table).filter_by(uuid=uuid).update(kwargs)
                sess.commit()
            return uuid
        else:
            # Else, add entry to database and generate new uuid
            if self.verbose:
                logger.info(f"~ Adding new entry for {kwargs['pid']} in {table} ~")
            uuid_kwargs = {
                **kwargs,
                'uuid': str(uid.uuid4()) if unique_id is None else unique_id
            }
            obj = insert(table).values(**uuid_kwargs)
            with self.session as sess:
                sess.execute(obj)
                sess.commit()
            return uuid_kwargs['uuid']

    #~ === BONE MASKS + MUSCLE MASKS IF EXIST ===
    def generate_mask_directories(self, type_):
        #~ Create directory structure for masks
        self.type_to_directory = {
            'bone': self.bone_mask_dir,
            'sm': self.sm_mask_dir,
            'imat': self.imat_mask_dir,
            'sf': self.sf_mask_dir,
            'vf': self.vf_mask_dir,
            'body': self.body_mask_dir,
            'vertebra': self.vertebra_mask_dir
            }

        with self.session as sess:
            #* Get all images from outputs table
            all_images = sess.query(self.tables.images).all()

        for row in all_images:
            row = dotdict(row._mapping)
            if type_ == 'bone':
                #* Load image and generate bone mask
                Image = self.loader_function(row.path_to_im)
                Image, _ = self.reorient(Image, orientation='RPI')
                Image = sitk.Cast(Image, sitk.sitkFloat32)
                Image = Image*float(row.offset)
                if row.modality == 'CBCT':
                    #* Resample CBCT to PCT and extract slice
                    with self.session as sess:
                        conditions = {'pid': row.pid, 'modality': 'CT'}
                        
                        pct_row = sess.query(self.tables.images).filter_by(**conditions).first()
                    pct = self.loader_function(pct_row.path_to_im)
                    pct, _ = self.reorient(pct, orientation='RPI')
                    Image = self.resample_cbct2pct(pct, Image)

                self.check_mask_directories(Image, row, type_)
                self.add_masks_to_database(type_, row)
            else:
                for level in self.v_level:
                    self.check_mask_directories(None, row, type_, level)
                    self.add_masks_to_database(type_, row, level=level)
                
    
    def check_mask_directories(self, Image, row, type_, level=None):
        #~ Check if masks exist, generate them if not
        if type_ == 'sm':
            root_dir = self.type_to_directory['sm'][level]
        elif type_ == 'imat':
            root_dir = self.type_to_directory['imat'][level]
        elif type_ == 'sf':
            root_dir = self.type_to_directory['sf'][level]
        elif type_ == 'vf':
            root_dir = self.type_to_directory['vf'][level]
        elif type_ == 'body':
            root_dir = self.type_to_directory['body'][level]
        else:
            root_dir = self.type_to_directory[type_]

        if row.series_uuid and row.study_uuid:
            path = os.path.join(root_dir, row.pid, row.study_uuid, row.series_uuid)
        else:
            pid, modality, acq_date = row.pid, row.modality, row.acquisition_date
            #* Define path
            if all(v is not None for v in [pid, modality, acq_date]):
                path = os.path.join(root_dir, pid, modality, acq_date)
            elif acq_date is None:
                path = os.path.join(root_dir, pid, modality)
            elif all(v is None for v in [modality, acq_date]):
                path = os.path.join(root_dir, pid)
            else:
                raise ValueError('Failed to create mask directories, unknown parameters for row: ', row) 

        #~ Check here
        if os.path.isdir(path) and 'mask.nii.gz' in os.listdir(path) and not self.regenerate_bones:
            #* Path and mask already exist, no need to generate
            if self.verbose:
                if type_ != 'bone':
                    logger.info(f"Detected {type_} mask for patient: {row.pid} - modality: {row.modality} - date: {row.acq_date} @ level {level}")    
                else:
                    logger.info(f"Detected {type_} mask for patient: {row.pid} - modality: {row.modality} - date: {row.acq_date}")
            return
        else: #* New patient detected, so updated directories & generate bone mask
            os.makedirs(path, exist_ok=True)
            if type_ == 'bone':
                #* Bone mask, if required.
                if self.verbose:
                    logger.info(f"Generating {type_} mask for patient: {row.pid} - modality: {row.modality} - date: {row.acq_date}")
               
                Bone = self.generate_bone_mask(Image, ast.literal_eval(row.spacing), threshold=self.bone_threshold, radius=self.bone_radius)
                Bone = sitk.Cast(Bone, sitk.sitkInt8)
                #bone_mask = sitk.GetArrayFromImage(Bone).astype(np.int8)
                #* Save mask
                sitk.WriteImage(Bone, os.path.join(path, 'mask.nii.gz'))
                #np.savez_compressed(os.path.join(path, 'mask.npz'), mask=bone_mask)

    def generate_bone_mask(self, Image, pixel_spacing, threshold = 350, radius = 3):
        #~ Create bone mask (by thresholding) for handling partial volume effect
        #@threshold in HU; radius in mm.
        logger.info(f"Generating bone mask using threshold ({threshold}) and expanding isotropically by {radius} mm")
        #* Apply threshold
        bin_filt = sitk.BinaryThresholdImageFilter()
        bin_filt.SetOutsideValue(1)
        bin_filt.SetInsideValue(0)

        if self.wm_mode:
            bin_filt.SetLowerThreshold(0)
            bin_filt.SetUpperThreshold(threshold+1024)
        else:
            bin_filt.SetLowerThreshold(-1024)
            bin_filt.SetUpperThreshold(threshold)

        bone_mask = bin_filt.Execute(Image)
        
        #* Convert to pixels
        pix_rad = [int(radius//elem) for elem in pixel_spacing]
        
        #* Dilate mask
        dil = sitk.BinaryDilateImageFilter()
        dil.SetKernelType(sitk.sitkBall)
        dil.SetKernelRadius(pix_rad)
        dil.SetForegroundValue(1)
        return dil.Execute(bone_mask)

    def add_masks_to_database(self, type_, row, level=None):
        type_to_directory = {
            'bone': self.bone_mask_dir,
            'sm': self.sm_mask_dir,
            'imat': self.imat_mask_dir, 
            'sf': self.sf_mask_dir,
            'vf': self.vf_mask_dir,
            'body': self.body_mask_dir,
            'vertebra': self.vertebra_mask_dir
            }
        assert type_ in type_to_directory, "Unknown mask directory"

        if type_ not in ['bone', 'vertebra']:
            root_dir = type_to_directory[type_][level]
        else:
            root_dir = type_to_directory[type_]
        
        logger.info(f"Adding masks of type: {type_} to database.")

        if row.series_uuid and row.study_uuid:
            path = os.path.join(root_dir, row.pid, row.study_uuid, row.series_uuid, 'mask.nii.gz')
            kwargs = {
                'pid': row.pid,
                'series_uuid': row.series_uuid,
                'study_uuid': row.study_uuid
            }

        else:
            pid, modality, acq_date = row.pid, row.modality, row.acquisition_date
            #* Define path
            if all(v is not None for v in [pid, modality, acq_date]):
                path = os.path.join(root_dir, pid, modality, acq_date, 'mask.nii.gz')
            elif acq_date is None:
                path = os.path.join(root_dir, pid, modality, 'mask.nii.gz')
            elif all(v is None for v in [modality, acq_date]):
                path = os.path.join(root_dir, pid, 'mask.nii.gz')
            else:
                raise ValueError('Failed to create mask directories, unknown parameters for row: ', row)
            kwargs = {
                'pid': pid,
                'modality': str(modality),
                'acquisition_date': str(acq_date)}

        with self.session as sess:
            if type_ == "bone":
                sess.query(self.tables.outputs).filter_by(
                    **kwargs).update({"path_to_bone_masks": path})
            sess.commit()

    #~ === SUPPLEMENTARY INPUTS ====
    def read_external_slice_numbers(self):
        filepath = os.path.join(self.ext_inputs, self.ext_slice_numbers)
        df = pd.read_csv(filepath)
        df['patient_id'] = df['patient_id'].astype(str)
        for pid in self.names:
            if pid in pd.unique(df['patient_id']):
                #* If PID in CSV, update entry
                entry = df.loc[(df['patient_id'] == pid)]
            elif pid+self.file_extension in pd.unique(df['patient_id']):
                #* If PID in CSV, update entry
                entry = df.loc[(df['patient_id'] == pid+self.file_extension)]
            else:
                if self.verbose:
                    print(Style.DIM + f'No entry in {self.ext_slice_numbers} for {pid}')
                continue
            if 'series_uid' in entry and 'study_uid' in entry:
                cond = {
                        'pid': pid,
                        'series_uuid': entry['series_uid'].values[0].strip(),
                        'study_uuid': entry['study_uid'].values[0].strip()
                    }
            else:
                cond = {'pid': pid}
            with self.session as sess:
                row = sess.query(self.tables.images).filter_by(**cond).first()
                row = dotdict(row._mapping)


            for level in self.v_level:
                if math.isnan(entry[level].values[0]):
                    continue

                #? ========================================
                #? This doesn't always work... 
                if ast.literal_eval(row.flip_axis):
                    slice_num = ast.literal_eval(row.size)[-1] - int(entry[level].values[0]) - 1 #Since size starts at 1 but indexing starts at 0
                    logger.info(f"Patient {row.pid} has been flipped along Z. Slice #: {int(entry[level].values[0])} -> {slice_num}")
                else:
                    slice_num = int(entry[level].values[0])
                logger.info(f"Patient {row.pid}. Slice #: {slice_num}")
                #? ==========================================

                # Correct for selected slice being the last one
                if slice_num == ast.literal_eval(row.size)[-1]:
                    logger.info("Selected slice - 1 since last slice selected")
                    slice_num -= 1
                
                with self.session as sess:
                    #* Find entries by PID and add slice_number
                    sess.query(self.tables.outputs).filter_by(
                        **cond).update({level: str(slice_num)})
                    sess.commit()
    
    def read_external_offset(self):
        filepath = os.path.join(self.ext_inputs, self.ext_offset)
        df = pd.read_csv(filepath)
        for pid in self.names:
            if pid in pd.unique(df['patient_id']):
                entry = df.loc[(df['patient_id'] == pid)]
                for i, row in entry.iterrows():
                    modality, acq_date, offset = row['Modality'], \
                        row['Acquisition_Date'], row['Offset']
                    kwargs = {
                        'pid': pid,
                        'modality': str(modality),
                        'acquisition_date': str(acq_date) if str(acq_date) != "None" else None
                    }
                    with self.session as sess:
                        sess.query(self.tables.images).filter_by(
                            **kwargs).update({"offset": str(offset)})
                        sess.commit()
            else:
                if self.verbose:
                    print(Style.DIM + f'No entry in {self.ext_offset} for {pid}')
                continue
    
   
    def _get_modalities(self):
        #~ Get distinct modalities in dataset, to apply correct model
        with self.session as sess:
            return [v.modality for v in sess.query(self.tables.images.c.modality).distinct()]

    #~ === STATIC METHODS ===
    @staticmethod
    def load_numpy(path):
        img = np.load(path)
        return sitk.GetImageFromArray(img)

    @staticmethod
    def load_nifty(path):
        #* Read nii volume
        return sitk.ReadImage(path)
    
    @staticmethod
    def load_dcm(path):
        #* Read DICOM directory
        reader = sitk.ImageSeriesReader()
        dcm_names = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(dcm_names)
        return reader.Execute()

    @staticmethod
    def reorient(Image, orientation='RPI'):
        orient = sitk.DICOMOrientImageFilter()
        orient.SetDesiredCoordinateOrientation(orientation)
        return orient.Execute(Image), orient
    
    @staticmethod
    def resample_cbct2pct(pct, cbct):
        return sitk.Resample(cbct, pct, sitk.Transform(), sitk.sitkLinear, 0, pct.GetPixelID())

    @staticmethod
    def resample_isotropic_grid(Image, pix_spacing=(1, 1, 1)):
        #* Resampling to isotropic grid
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetOutputDirection(Image.GetDirection())
        resample.SetOutputOrigin(Image.GetOrigin())
        ratio = tuple([x/y for x, y in zip(Image.GetSpacing(), pix_spacing)])
        new_size = [np.round(x*s) \
            for x, s in zip(Image.GetSize(), ratio)]
        resample.SetSize(np.array(new_size, dtype='int').tolist())
        resample.SetOutputSpacing(pix_spacing)
        #* Ratio flipped to account for sitk.Image -> np.array transform
        return resample.Execute(Image), ratio[::-1]

    @staticmethod
    def get_image_attributes(Image):
        dict_ = {'spacing': Image.GetSpacing(), 
                'direction': Image.GetDirection(),
                'size': Image.GetSize(),
                'origin': Image.GetOrigin()}
        return {k: str(v) for k, v in dict_.items()}

    @staticmethod
    def wl_norm(img, window, level):
        minval = level - window/2
        maxval = level + window/2
        wld = np.clip(img, minval, maxval)
        wld -= minval
        wld /= window
        return wld

    @staticmethod
    def _get_window_level(v_level, modality):
        #~ Query settings for specific models (window/level)
        settings_bank = {
            'C3': {
                'CT': {'window': 400, 'level': 50}, 
                'CBCT': {'window': 600, 'level': 76}
                },
            'T4': {
                'CT': {'window': 400, 'level': 50}
                },
            'T9': {
                'CT': {'window': 400, 'level': 50}
                },
            'T12': {
                'CT': {'window': 400, 'level': 50}
                },
            'L3': {
                'CT': {'window': 400, 'level': 50}
                },
            'L5': {
                'CT': {'window': 400, 'level': 50}
                },
            'Thigh': {
                'CT': {'window': 400, 'level': 50}
            }
        }

        if v_level in settings_bank:
            if modality in settings_bank[v_level]:
                return settings_bank[v_level][modality]
            else:
                raise ValueError(f'No {modality} model for {v_level}.')
        else:
            raise ValueError(f'Model for {v_level} not implemented yet.')