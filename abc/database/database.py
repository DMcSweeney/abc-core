

import os
import logging
import sqlalchemy as sa
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
import ast
from datetime import datetime

from colorama import Fore, Style, Back

from .tables import Base, Tables
from .base import DataBase_Helper
from ..mixin import dotdict

VERTEBRAL_LEVELS = ['C3', 'T4', 'T9', 'T12', 'L3', 'L5', 'Thigh']

logger = logging.getLogger(__name__)

class Database(DataBase_Helper):
    def __init__(self, args, crop_size=(256, 256), output_shape=(128, 128, 128)):

        #* Directories
        self.input_dir = args.input_dir
        self.output_dir = args.output_dir
        self.v_level = args.vertebral_levels
        os.makedirs(self.output_dir, exist_ok=True)
        #* Make sub-directories in output folder
        self.sm_mask_dir = {v:
            os.path.join(self.output_dir, f'masks/skeletal_muscle/{v}/') for v in self.v_level}
        self.sf_mask_dir = {v:
            os.path.join(self.output_dir, f'masks/subcutaneous_fat/{v}/') for v in self.v_level}
        self.vf_mask_dir = {v:
            os.path.join(self.output_dir, f'masks/visceral_fat/{v}/') for v in self.v_level}
        self.imat_mask_dir = {v:
            os.path.join(self.output_dir, f'masks/IMAT/{v}/') for v in self.v_level}
        self.body_mask_dir = {v:
            os.path.join(self.output_dir, f'masks/body/{v}/') for v in self.v_level}

        self.vertebra_mask_dir = os.path.join(self.output_dir, 'masks/vertebrae/')

        self.bone_mask_dir = os.path.join(self.output_dir, 'masks/bone_masks/')
        self.bone_threshold = args.bone_threshold
        self.bone_radius = 5 # Expand bone mask by 3mm 
        self.regenerate_bones = args.regenerate_bones # If yes, will overwrite bone masks

        #* Attributes
        self.verbose = args.print_to_console
        self.modality = args.default_modality #! Default modality when not known
        self.wm_mode = args.wm_mode
        
        self.accepted_modalities = ['CT', 'CBCT', 'PCT']
        self.crop_size = crop_size
        self.output_shape = output_shape
        self.num_slices = args.num_slices
        
        #* SUPPLEMENTARY_INPUTS
        self.ext_inputs = args.ext_inputs
        #* External slice numbers
        self.ext_slice_numbers = args.ext_slice_numbers if args.ext_inputs is not None else None
        self.ext_offset = args.ext_offset if args.ext_inputs is not None else None
        self.restrict_to_csv = args.restrict_to_csv

        if self.restrict_to_csv and self.ext_slice_numbers is not None:
            logger.warning(Fore.RED + "!! - ONLY PERFORMING INFERENCE ON PATIENTS IN CSV FILE - !!" + Style.RESET_ALL)

        #* Init. connection to database
        self.path_to_db = args.path_to_db
        self.engine = sa.create_engine(f"sqlite:///{self.path_to_db}",
                echo=False, future=True)

        self.Base = Base
        self.tables = Tables(VERTEBRAL_LEVELS)
        self.session = Session(self.engine)
        
        
        #* If file doesn't exist, create database 
        if not os.path.isfile(self.path_to_db):
            self.Base.metadata.create_all(self.engine)

    def _clear_database(self):
        #~ Deletes database file
        os.remove(self.path_to_db)

    def _init_database(self):
        #~ Add elements to database if empty
        with self.session as sess:
            #* All names with images in input dir.
            self.names = [v.pid for v in \
                sess.query(self.tables.images.c.pid).distinct()]
            self.uuids = [v.uuid for v in \
                sess.query(self.tables.images.c.uuid).distinct()]
        #* Check supplementary inputs
        if self.ext_inputs is not None:
            self._check_externals()

    def _check_externals(self):
        #~ ======= SLICE NUMBERS CSV =======
        if self.ext_slice_numbers is not None:
            logger.info('----------- External CSV file detected with slice numbers -----------')
            self.read_external_slice_numbers()

        #~====== CBCT OFFSET ==============
        if self.ext_offset is not None:
            logger.info('----------- External CSV file detected with CBCT offset -----------')
            self.read_external_offset()

    def _generate_masks(self):
        #~===== SEGMENTATION MASK===========
        logger.info("Generating output directories")
        for dir_ in self.sm_mask_dir.values():
            os.makedirs(dir_, exist_ok=True)
        for dir_ in self.sf_mask_dir.values():
            os.makedirs(dir_, exist_ok=True)
        for dir_ in self.vf_mask_dir.values():
            os.makedirs(dir_, exist_ok=True)

        self.generate_mask_directories(type_='sm')
        self.generate_mask_directories(type_='imat')
        self.generate_mask_directories(type_='sf')
        self.generate_mask_directories(type_='vf')
        self.generate_mask_directories(type_='body')
        #~ === Vertebrae directory for spine labelling
        self.generate_mask_directories(type_='vertebra')

        #~========== BONE MASKS ============
        os.makedirs(self.bone_mask_dir, exist_ok=True)
        self.generate_mask_directories(type_='bone')


    def _collect_inputs(self):
        #~ Choose a data directory + load data
        #! Assumes all files in dir. are same format
        self.files = os.listdir(self.input_dir)
        self.num_files = len(self.files)
        #* Check if all elements in self.files are directories
        dir_test = [os.path.isdir(os.path.join(self.input_dir, elems))\
             for elems in self.files]
        
        #* If directory... 
        if all(dir_test):
            self.get_file_extension_from_directory(self.input_dir)
            #* This helps determine how patient ID is defined
            #* I.e. directory name vs filename
            self.from_directory = True 

        #* If no directories, assume files and check extension
        elif not all(dir_test):
            if self.files[0].endswith('.nii'):
                self.file_extension = ".nii"
                self.loader_function = self.load_nifty
                self.from_directory = False
            elif self.files[0].endswith('.nii.gz'):
                self.file_extension = ".nii.gz"
                self.loader_function = self.load_nifty
                self.from_directory = False
            elif self.files[0].endswith('.npy'):
                self.file_extension = '.npy'
                self.loader_function = self.load_numpy
                self.from_directory = False
            else:
                self.file_extension = self.files[0].split('.')[-1].strip()
                raise ValueError(f'{self.file_extension} not supported.')
        else: 
            raise ValueError('Input directory structure not supported.')

        #* IF an external slice number CSV has been provided 
        #* & restrict_to_csv is True, update self.files to only include patients in CSV
        if self.restrict_to_csv and self.ext_slice_numbers is not None:
            filepath = os.path.join(self.ext_inputs, self.ext_slice_numbers)
            df = pd.read_csv(filepath)
            names = [str(x) for x in pd.unique(df['patient_id']) if str(x) in self.files or str(x)+self.file_extension in self.files]
            #* Check if names in CSV include file extension
            ext_test = [name.endswith(self.file_extension) for name in names]
            if not self.from_directory or all(ext_test):
                self.files = [x.rstrip(self.file_extension) if self.file_extension in x else x for x in names]
            else:
                #* If directory assume no filename
                self.files = [x for x in self.files if x in names]
            
    def _populate_database(self):
    #~ Cross directory structure and update database with files
        for file in self.files:
            #! file is file or directory if dcm
            if self.from_directory and self.file_extension == '.dcm':
                metadata = self.parse_dicom_directory(file)
                self.add_pid_to_db(**metadata)

            elif self.from_directory and self.file_extension != '.dcm':
                metadata = self.parse_directory(file)
                # Get number of elements
                num_ims = set([len(val) for val in metadata.values()])
                assert len(num_ims)==1, print("Args are mismatched, can't update db. Size: ", len(num_ims))
                metadata['pid'] = [file] * next(iter(num_ims))
                self.add_pid_to_db(**metadata)

            else: 
                path = os.path.join(self.input_dir, file)
                pid = file.strip(self.file_extension)
                metadata = {'path': [path], 'pid': [pid]}
                self.add_pid_to_db(**metadata)


