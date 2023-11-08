"""
Inference engines
#TODO:
    - Optimize ONNX
"""
from asyncio.base_subprocess import ReadSubprocessPipeProto
import os
import resource
import time
import numpy as np
import onnxruntime as ort
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import logging

import sqlalchemy as sa
from .writer import segWriter as Writer
from colorama import Style, Fore
import ast
import SimpleITK as sitk
from ...mixin import dotdict

import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from monai.transforms.spatial.functional import resize
from scipy.special import softmax
import skimage

logger = logging.getLogger(__name__)


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

class segEngine():
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.v_level = dataset.v_level
        self.verbose = args.print_to_console

        self.num_slices = args.num_slices #* Number of slices to segment either side of reference (default:0)
        self.workers = self.num_slices*2 + 1
        if self.workers > 10: # Max at 10 workers
            self.workers=10
        self.wm_mode = args.wm_mode
        self.normalise = args.normalise


        self.regen_masks = args.regen_masks
        self.output_dir = args.output_dir

        self.db = dataset.db

        self.modalities = self._get_modalities() #* Get modalities from database 
        self._init_model_bank() #* Load bank of models
        self._set_options() #* Set ONNX session options

        self.ort_sessions = {mod: ort.InferenceSession(
            self.model_paths[mod], sess_options=self.sess_options) for mod in self.modalities}
        #Pre-processing transforms
        self.transforms = A.Compose([
            #A.Resize(height=512, width=512),
            A.Normalize(mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225), max_pixel_value=1),
            ToTensorV2()
            ],)
        


    def forward(self):
        #~ Run inference
        for modality in self.modalities:
            logger.info(f"Inference on {modality} scans")
            self.forward_pass(modality)

    def forward_pass(self, modality):
        #~ Iterate over all images
        t = time.time()
        mod_dataset = self.dataset # Create instance of dataset, then update modality and use that for dataloader
        mod_dataset.modality = modality
        self.modality = modality
        writer = Writer(self, modality, mod_dataset) #* init. writer

        for data in tqdm(mod_dataset, desc=f'Segmentation Inference for {modality} scans @ {self.v_level}'):
            if data is None: #* No images for this patient
                continue
            #* Data contains dataframe with image info.
            for _, row in data['images'].iterrows():
                row = dotdict(row)
                logger.info(f"Patient ID: {row.pid} - Scan uuid: {row.uuid}")
                #* Check if muscle mask exists, else segment
                with self.db.session as sess:
                    #* Should only be one match since querying on uuid
                    kwargs = {'uuid': row.uuid}
                    out_row = sess.query(self.db.tables.outputs).filter_by(
                        **kwargs).first()
                    out_row = dotdict(out_row._mapping)
                out_path = out_row.path_to_bone_masks.replace('bone_masks', f'skeletal_muscle/{self.v_level}')
                if not self.regen_masks and os.path.isfile(out_path):
                    logger.warning(f"This mask already exists and you have asked not to regenerate masks - skipping")
                    continue

                Image, slice_number = self.load_image(row) # Returns SimpleITK image and reference slice
                image = sitk.GetArrayFromImage(Image)
                
                
                logger.info("Reading bone mask, this will be removed from the prediction")
                Bone = sitk.ReadImage(out_row["path_to_bone_masks"]) if os.path.isfile(out_row["path_to_bone_masks"]) else None
                self.bone = np.logical_not(sitk.GetArrayFromImage(Bone)) if Bone is not None else None

                holder = np.zeros_like(image, dtype=np.int8)

                #* Create some holders to put predictions
                self.holders = {
                    'skeletal_muscle': holder.copy(),
                    'subcutaneous_fat': holder.copy(), 
                    'visceral_fat': holder.copy(),
                    'IMAT': holder.copy(),
                    'body': holder.copy()
                }

                #* Iterate across the reference slice
                self.idx2slice = {}
                logger.info(f"Pre-processing input slices (# slices: {self.num_slices*2 +1})")
                for i, slice_ in enumerate( np.arange(slice_number-self.num_slices, slice_number+self.num_slices+1) ):
                    im_tensor = self.pre_process(image, slice_)

                    if im_tensor is None:
                        logger.error(f"The selected slice ({slice_number}) is out of range for patient ({row.pid}) - skipping.")
                        continue

                    if i == 0:
                        img = im_tensor[None]
                    else:
                        img = torch.cat([img, im_tensor[None]], axis=0)
                    self.idx2slice[i] = slice_

                ### REMOVING THIS --- RESIZING INPUTS AND BONE MASK INSTEAD
                # ## Check if dimensions aren't divisible by 2
                # is_divisible = [True if x%2 == 0 else False for x in img.shape[-2:] ]
                # if not all(is_divisible):
                #     logger.error("Input size is not divisible by 2 - skipping. Add resizing to transforms?")
                #     continue
                # is_too_small = [True if x >256 else False for x in img.shape[-2:]]
                # if not all(is_too_small):
                #     logger.error(f"Image is too small, will. have issues in model. Add resizing to transforms? Size: {img.shape}")
                #     continue

                self.img = np.array(img)
                
                #* Holders for channel 2 and channel 3 outputs
                chan2_lst = []
                chan3_lst = []
                logger.info(f"==== INFERENCE ID: {row.pid}====")
                for i in range(img.shape[0]):
                    chan1, chan2, chan3 = self.per_slice_inference(i)
                    chan2_lst.append(chan2)
                    chan3_lst.append(chan3)

                #* Convert predictions back to ITK Image, using input Image as reference
                logging.info(f"Generating IMAT mask using thresholds: {mod_dataset.fat_threshold}")
                blurred_image = skimage.filters.gaussian(image, sigma=0.7, preserve_range=True)
                fat_threshold = np.logical_and(
                    blurred_image >= mod_dataset.fat_threshold[0],
                    blurred_image <= mod_dataset.fat_threshold[1]
                    ).astype(np.int8)
                IMAT = np.logical_and(fat_threshold, self.holders['skeletal_muscle']).astype(np.int8)
                IMAT = skimage.measure.label(IMAT, connectivity=2, return_num=False) # connected components

                #* Remove IMAT from muscle mask
                skeletal_muscle = np.where(IMAT == 1, 0, self.holders['skeletal_muscle']).astype(np.int8)
                logger.info(f"Converting IMAT mask to ITK Image. Size: {IMAT.shape}")
                IMAT = self.npy2itk(IMAT, Image)

                logger.info(f"Converting skeletal muscle mask to ITK Image. Size: {skeletal_muscle.shape}")
                SkeletalMuscle = self.npy2itk(skeletal_muscle, Image)
                logger.info("Writing skeletal muscle mask")
                self.save_prediction(row.uuid, 'skeletal_muscle', SkeletalMuscle)
                logger.info("Writing IMAT mask")
                self.save_prediction(row.uuid, 'IMAT', IMAT)

                #* Repeat with subcut and visceral fat. if they exist
                if any(chan2_lst) and any(chan3_lst):
                    logger.info(f"Converting subcutaneous fat mask to ITK Image. Size: {self.holders['subcutaneous_fat'].shape}")
                    SubcutaneousFat = self.npy2itk(self.holders['subcutaneous_fat'], Image)
                    logger.info(f"Converting visceral fat mask to ITK Image. Size: {self.holders['visceral_fat'].shape}")
                    VisceralFat = self.npy2itk(self.holders['visceral_fat'], Image)
                    logger.info("Writing subcutaneous fat mask")
                    self.save_prediction(row.uuid, 'subcutaneous_fat', SubcutaneousFat)
                    logger.info("Writing visceral fat mask")
                    self.save_prediction(row.uuid, 'visceral_fat', VisceralFat)
                elif any(chan2_lst) and not any(chan3_lst):
                    logger.info(f"Converting body mask to ITK Image. Size: {self.holders['body'].shape}")
                    Body = self.npy2itk(self.holders['body'], Image)
                    logger.info("Writing body mask")
                    self.save_prediction(row.uuid, 'body', Body)
                else:
                    logger.warning(f"No predictions other than skeletal muscle")
           
        logging.info(f'Execution time (s): {np.round(time.time() - t, 2)} for {mod_dataset.__len__()} examples.')
        

    def load_image(self, row):
        
        with self.db.session as sess:
            #* Get the slice number for this image
            res = sess.query(self.db.tables.outputs).filter_by(uuid=row.uuid).first()
            res = dotdict(res._mapping)

        #* Load image and reorient
        Image = self.db.loader_function(row.path_to_im)
        Image = self.reorient(Image, orientation='RPI')
        if self.modality == 'CBCT':
            #* Resample CBCT to PCT and extract slice
            with self.db.session as sess:
                conditions = {'pid': row.pid, 'modality': 'CT'}
                row = sess.query(self.db.tables.images).filter_by(**conditions).first()
            pct = self.db.loader_function(row.path_to_im)
            pct = self.reorient(pct, orientation='RPI')
            Image = self.resample_cbct2pct(pct, Image)
        return Image, int(res[self.v_level])

    def pre_process(self, image, slice_number):
        #* Pre-processing
        try:
            im = image[slice_number]
        except IndexError: #* Slice out of bounds
            return None
        
        if self.wm_mode:
            logging.info("Applying worldmatch correction (-1024 HU)")
            im -= 1024
        if self.normalise:
            self.settings = self._get_window_level()
            logging.info(f"Window/Level ({self.settings['window']}/{self.settings['level']}) normalisation")
            im = self.wl_norm(im, window=self.settings['window'], level=self.settings['level'])
        
        logging.info(f"Converting input to three channels")
        im = self.expand(im) #* 3 channels
        logging.info(f"Applying transforms: {self.transforms}")
        augmented = self.transforms(image=im)
        return augmented['image']

    def per_slice_inference(self, i):
        input = self.img[i] 
        is_divisible = [True if x % 2 == 0 else False for x in input.shape[-2:] ]
        is_too_small = [True if x < 256 else False for x in input.shape[-2:]]
        if not all(is_divisible) or all(is_too_small): #TODO check this works
            logger.info(f"Model input size: {input.shape} detected, resampling to 512x512")
            input = resize(input, (512, 512), mode='bicubic')
        prediction = self.inference(input)
        logger.info(f"Splitting prediction (shape: {prediction.shape}) into compartments.")
        chan1, chan2, chan3 = self.split_predictions(prediction) # Split prediction into compartments#
    
        self.holders['skeletal_muscle'][self.idx2slice[i]] = self.remove_bone(i, chan1) if self.bone is not None else chan1

        if chan2 is not None and chan3 is not None:
            logger.info("Fat segmentations detected - adding")
            self.holders['subcutaneous_fat'][self.idx2slice[i]] = self.remove_bone(i, chan2) if self.bone is not None else chan2
            self.holders['visceral_fat'][self.idx2slice[i]] = self.remove_bone(i, chan3) if self.bone is not None else chan3
            return True, True, True # Muscle/SF/VF
        
        elif chan2 is not None and chan3 is None: #Muscle/Body
            logger.info("Body mask detected - adding")
            self.holders['body'][self.idx2slice[i]] = self.remove_bone(i, chan2) if self.bone is not None else chan2
            return True, True, False

        return True, False, False

    def remove_bone(self, i, pred):
        # Resize to match prediction/input
        bone_mask = resize(self.bone[self.idx2slice[i]], (512, 512), mode='nearest')
        return np.logical_and(pred, bone_mask)

    def inference(self, img):
        #* Forward pass through the model
        t= time.time()
        ort_inputs = {self.ort_sessions[self.modality].get_inputs()[0].name: \
            img.astype(np.float32)}
        logging.info(f'Model load time (s): {np.round(time.time() - t, 7)}')
        #* Inference
        t= time.time()
        outputs = np.array(self.ort_sessions[self.modality].run(None, ort_inputs)[0])
        outputs = np.squeeze(outputs)
        logging.info(f'Inference time (s): {np.round(time.time() - t, 7)}')
        logging.info(f"Model outputs: {outputs.shape}")

        if outputs.shape[0] in [3, 4]:
            logging.info("Multiple channels detected, applying softmax")
            pred = np.argmax(softmax(outputs, axis=0), axis=0).astype(np.int8) # Argmax then one-hot encode
            preds = [np.where(pred == val, 1, 0) for val in np.unique(pred)] # one-hot encode
            return np.stack(preds)
        else:
            logging.info("Single channel detected, applying sigmoid")
            return np.round(self.sigmoid(outputs)).astype(np.int8)
        

    def split_predictions(self, pred):
        #* Split into components
        if pred.shape[0] == 4: #background,muscle,sub,visc
            return pred[1], pred[2], pred[3]
        elif pred.shape[0] == 3: #background, muscle, body mask
            return pred[1], pred[2], None
        else:
            return pred[1], None, None

    def save_prediction(self, uuid, tag, Prediction):
        with self.db.session as sess:
            #* Should only be one match since querying on uuid
            kwargs = {'uuid': uuid}
            out_row = sess.query(self.db.tables.outputs).filter_by(
                **kwargs).first()
            out_row = dotdict(out_row._mapping)

        #* Save mask to outputs folder
        out_path = out_row.path_to_bone_masks.replace('bone_masks', f'{tag}/{self.v_level}')
        logger.info(f"Saving prediction. Scan uuid: {uuid}. Path: {out_path}")
        sitk.WriteImage(Prediction, out_path)
            
        #* Add path to database
        with self.db.session as sess:
            update = {f"{self.v_level}_path_to_{tag}_masks": out_path}
            sess.query(self.db.tables.outputs).filter_by(
                **kwargs).update(update)
            sess.commit()

    
    #~  ================= OPTIONS ==================
    def _init_model_bank(self):
        #* Paths to segmentation models

        if getattr(sys, 'frozen', False):
            model_bank = {
                'C3': {'CT': resource_path('stk_viewer/stk/segmentation/model/onnx_models/c3_pCT.quant.onnx'),
                        'CBCT': resource_path('stk_viewer/stk/segmentation/model/onnx_models/C3_cbct.onnx')},
                'T4': {'CT': resource_path('stk_viewer/stk/segmentation/model/onnx_models/t4_pet.quant.onnx')},
                'T9': {'CT': resource_path('stk_viewer/stk/segmentation/model/onnx_models/t9_pet.quant.onnx')},
                'T12': {'CT':
                 #resource_path('stk_viewer/stk/segmentation/model/onnx_models/T12_COCO_18pats.quant.onnx')
                 resource_path('stk_viewer/stk/segmentation/model/onnx_models/shufflenetV2-T12-M.quant.onnx')
                 },
                'L3': {'CT':# resource_path('stk_viewer/stk/segmentation/model/onnx_models/L3_COCO_132pats.quant.onnx')},
                       resource_path('stk_viewer/stk/segmentation/model/onnx_models/shufflenetV2-L3-FM.quant.onnx')
                },
                'L5': {'CT':resource_path('stk_viewer/stk/segmentation/model/onnx_models/TitanMixNet-Med-L5-FM.onnx')
                       },
                'Thigh': {'CT': resource_path('stk_viewer/stk/segmentation/model/onnx_models/Thigh_14pats.quant.onnx')}
                }
        else:
            model_bank = {
                'C3': {'CT': './models/segmentation/c3_pCT.quant.onnx',
                        'CBCT': './models/segmentation/C3_cbct.onnx'},
                'T4': {'CT': #'./models/segmentation/shufflenetV2-T4-M.quant.onnx'
                    './models/segmentation/TitanMixNet-Med-T4-Body-M.onnx'
                },
                'T9': {'CT': #'./models/segmentation/shufflenetV2-T9-M.quant.onnx'
                    './models/segmentation/TitanMixNet-Med-T9-Body-M.onnx'
                },
                'T12': {'CT': #'./models/segmentation/shufflenetV2-T12-M.quant.onnx'
                        './models/segmentation/TitanMixNet-Med-T12-Body-M.onnx'
                        },
                'L3': {'CT': #'./models/segmentation/L3_COCO_132pats.quant.onnx'},
                       './models/segmentation/TitanMixNet-Med-L3-FM.onnx'
                },
                'L5': {'CT': './models/segmentation/TitanMixNet-Med-L5-FM.onnx'
                       },
                'Thigh': {'CT': './models/segmentation/Thigh_14pats.quant.onnx'}
                }
        
        if self.v_level in model_bank:
            self.model_paths = {}
            for modality in self.modalities:
                if modality in model_bank[self.v_level]:
                    self.model_paths[modality] = model_bank[self.v_level][modality]
                else:
                    raise ValueError(f'No {modality} model for {self.v_level}.')
        else: 
            raise ValueError(f'Model for {self.v_level} not implemented yet.')

    def _get_window_level(self):
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

        if self.v_level in settings_bank:
            if self.modality in settings_bank[self.v_level]:
                return settings_bank[self.v_level][self.modality]
            else:
                raise ValueError(f'No {self.modality} model for {self.v_level}.')
        else:
            raise ValueError(f'Model for {self.v_level} not implemented yet.')

    def _set_options(self):
        #* Inference options
        self.sess_options = ort.SessionOptions()
        self.sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        self.sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # ! or ORT_PARALLEL
        self.sess_options.log_severity_level = 4
        self.sess_options.enable_profiling = False
        self.sess_options.inter_op_num_threads = os.cpu_count() - 1
        self.sess_options.intra_op_num_threads = os.cpu_count() - 1
    
    def _get_modalities(self):
        #~ Get distinct modalities in dataset, to apply correct model
        with self.db.session as sess:
            return [v.modality for v in sess.query(self.db.tables.images.c.modality).distinct()]

    #~ ==================== UTILS ====================
    @staticmethod
    def to_numpy(x):
        return x.detach().cpu().numpy() if x.requires_grad else x.cpu().numpy()

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    @staticmethod
    def reorient(Image, orientation='RPI'):
        orient = sitk.DICOMOrientImageFilter()
        orient.SetDesiredCoordinateOrientation(orientation)
        return orient.Execute(Image)
    
    @staticmethod
    def resample_cbct2pct(pct, cbct):
        return sitk.Resample(cbct, pct, sitk.Transform(), sitk.sitkLinear, 0, pct.GetPixelID()) 
    
    @staticmethod
    def expand(img):
        #* Convert to 3 channels
        return np.repeat(img[..., None], 3, axis=-1)
    
    @staticmethod
    def npy2itk(npy, reference):
        #* npy array to itk image with information from reference
        Image = sitk.GetImageFromArray(npy)
        Image.CopyInformation(reference)
        return Image
    
    @staticmethod
    def wl_norm(img, window, level):
        minval = level - window/2
        maxval = level + window/2
        wld = np.clip(img, minval, maxval)
        wld -= minval
        wld /= window
        return wld