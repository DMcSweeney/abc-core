"""
Calculates metrics and plots results
"""
import os
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import SimpleITK as sitk
import matplotlib.pyplot as plt
from datetime import datetime
import polars as pl

import pathos.multiprocessing as mp

from .mixin import dotdict

logger = logging.getLogger(__name__)

class Extractor():
    def __init__(self, database):
        self.database = database
        self.re_extract_stats = False #! This is only here to skip on large datasets

    def extract_stats(self, v_levels, thresholds):
        """
        Extract area/density for all levels and compartments
        Area reported in voxels. 
        """
        self.thresholds = thresholds
        for level in v_levels:
            self.level = level
            logger.info(f"Extracting statistics at {level}.")
            
            if level in ['L3']:
                del thresholds['body']

            for uuid in self.database.uuids:
                kwargs = {'uuid': uuid}
                logger.info(f"Scan uuid: {uuid}")
                #* Query the images and outputs tables with this uuid
                image_row, output_row = self.get_image_info(uuid)

                #* Check if relevant slice has been recorded
                if output_row[level] is None:
                    logging.error(f"No slice recorded for {image_row.pid} (uuid: {uuid})")
                    continue
                
                #* Filter what operations need to be done -- as it's most time consuming element
                thresholds_to_use = thresholds.copy()
                for tag in thresholds.keys():
                    #* Read predicted mask
                    if output_row[f"{level}_path_to_{tag}_masks"] is None:
                        logger.error(f"Can't find {tag} mask for patient: {image_row.pid}. Try to init. database/perform segmentation")
                        del thresholds_to_use[tag]
                        continue

                    #! Skipping if stats already extracted!!
                    if output_row[f"{level}_{tag}_area"] is not None and not self.re_extract_stats:
                        logger.warning(f"{level} {tag} area already extracted and re-extract stats is set to {self.re_extract_stats} - skipping")
                        del thresholds_to_use[tag]
                        continue

                    if output_row[f"{level}_{tag}_density"] is not None and not self.re_extract_stats:
                        logger.warning(f"{level} {tag} density already extracted and re-extract stats is set to {self.re_extract_stats} - skipping")
                        del thresholds_to_use[tag]
                        continue
                
                if not thresholds_to_use:
                    logger.warning(f"No new extractions to perform on scan: {uuid} - skipping")
                    continue

                #* Load, reorient (+offset and resample if CBCT)
                Image = self.prepare_image(image_row)
                
                #* Now iterate over compartments
                for tag in thresholds_to_use.keys():
                    self.tag = tag
                    logger.info(f"Calculating {tag} area/density")
                    

                    logger.info(f"Loading mask from: {output_row[f'{self.level}_path_to_{self.tag}_masks']}")
                    Prediction = sitk.ReadImage(output_row[f"{self.level}_path_to_{self.tag}_masks"])#
                    prediction = sitk.GetArrayFromImage(Prediction)
                    image = sitk.GetArrayFromImage(Image)

                    #* WM offset 
                    if self.database.wm_mode:
                        logger.info(f"Applying worldmatch correction")
                        image -= 1024
                    
                    #* Apply thresholding
                    if not all([x is None for x in thresholds_to_use[tag]]):
                        threshold_image = np.logical_and(
                            image >= thresholds_to_use[tag][0],
                            image <= thresholds_to_use[tag][1],
                        ).astype(np.int8)

                        prediction = np.logical_and(threshold_image, prediction).astype(np.int8)

                    #* Extract slice
                    slice_number = int(output_row[level])
                    logger.info(f"Extracting mean stats across {self.database.num_slices*2+1} slices")
                    slices =  [x for x in np.arange(slice_number-self.database.num_slices, slice_number+self.database.num_slices+1) ]
                    updates = []
                    for slice_ in slices:
                        update = self.per_slice_wrapper(image, prediction, output_row, slice_)
                        if update is None:
                            logger.error(f"The selected slice ({slice_number}) is out of range for patient ({image_row.pid}). Scan uuid: {uuid} - skipping.")
                            continue
                        updates.append(update)

                    keys = [f"{self.level}_{self.tag}_area", f"{self.level}_{self.tag}_density"]
                    
                    #* Returns mean & stdev across slices
                    update = {}
                    for key in keys:
                        update[key] = np.mean([i[key] for i in updates])
                        update[f'{key}_stdev'] = np.std([i[key] for i in updates]) 

                    # #* Update database
                    with self.database.session as sess:
                        sess.query(self.database.tables.outputs).filter_by(
                            **kwargs).update(update)
                        sess.commit()

    def per_slice_wrapper(self, image, prediction, output_row, slice_number):
        #*
        try:
            im = image[slice_number]
        except IndexError:
            return None

        pred = prediction[slice_number]
        #* Get bone mask
        if os.path.isfile(output_row["path_to_bone_masks"]):
            Bone = sitk.ReadImage(output_row["path_to_bone_masks"])
            bone = sitk.GetArrayFromImage(Bone)
            bone = np.logical_not(bone[slice_number])
        else:
            logger.warning(f"No bone mask detected.")
            bone=None

        update = self.stat_calculation(im, pred, bone)
        return update

    def stat_calculation(self, image, mask, bone=None):
        if bone is not None:
            mask = np.logical_and(mask, bone).astype(np.int8)

        #* Area calculation          
        area = float(np.sum(mask))
        #* Density calculation
        density = np.mean(image[mask==1])
        update = {f"{self.level}_{self.tag}_area": area,
                        f"{self.level}_{self.tag}_density": density}
        
        return update
        
    def prepare_image(self, image_row, type_='image'):
        #* Load image
        Image = self.database.loader_function(image_row.path_to_im)
        Image, _ = self.database.reorient(Image, orientation='RPI')
        
        if float(image_row.offset) != 1. and type_ == 'image':
            Image = sitk.Cast(Image, sitk.sitkFloat32)
            Image = Image*float(image_row.offset)
            Image = sitk.Cast(Image, sitk.sitkInt32)

        #* If CBCT, resample to PCT
        if image_row.modality == 'CBCT':
            logging.info("CBCT detected - Resampling to reference CT.")
            #~ If CBCT, resample to PCT and update slice num
            with self.session as sess:
                conditions = {'pid': image_row.pid, 'modality': 'CT'}
                pct_row = sess.query(self.tables.images).filter_by(**conditions).first()
            pct = self.loader_function(pct_row.path_to_im)
            pct, _ = self.reorient(pct, orientation='RPI')
            Image = self.resample_cbct2pct(pct, Image)
        return Image

    def get_image_info(self, uuid):
        with self.database.session as sess:
            kwargs = {'uuid': uuid}
            im_row = sess.query(self.database.tables.images).filter_by(
                **kwargs).first()
            out_row = sess.query(self.database.tables.outputs).filter_by(
                **kwargs).first()
            image_row = dotdict(im_row._mapping)
            output_row = dotdict(out_row._mapping)
        return image_row, output_row
    
    #** ================ WRITE RESULTS =============
    def stats_to_csv(self, output_path, tag):
        #~ Format muscle stats into a csv file
        df = pd.DataFrame(columns=['filename', 'modality', 
        'acquisition_date', 'area (voxels)', 'area (stdev)', 'density', 'density (stdev)', 'sanity_check'])
        i = 1
        #* Over vertebral levels (defaults to writing one file per level)
        for level in self.database.v_level:
            logger.info(f"Writing {tag} statistiscs at {level} to CSV file.")
            #* Iterate over files in input directory
            for i, uuid in enumerate(self.database.uuids):
                with self.database.session as sess:
                    row = sess.query(self.database.tables.outputs).filter_by(uuid=uuid).first()
                    im_row = sess.query(self.database.tables.images).filter_by(uuid=uuid).first()
                row = dotdict(row._mapping)
                im_row = dotdict(im_row._mapping)

                if row[f"{level}_path_to_{tag}_masks"] is None and tag != 'IMAT':
                    logger.error(f"No {tag} mask found for patient id: {im_row.pid}")
                    continue
                
                if self.database.file_extension == '.dcm':
                    filename = row.pid
                else:
                    filename = row.pid + self.database.file_extension
                
                area, density, sanity = row[f"{level}_{tag}_area"], row[f"{level}_{tag}_density"], row[f'{level}_sanity_check']
                area = float(area) if area is not None else np.nan
                density = float(density) if density is not None else np.nan
                # Get standard deviations
                area_stdev, density_stdev = row[f"{level}_{tag}_area_stdev"], row[f"{level}_{tag}_density_stdev"]
                area_stdev = float(area_stdev) if area_stdev is not None else np.nan
                density_stdev = float(density_stdev) if density_stdev is not None else np.nan

                date = pd.to_datetime(
                    row.acquisition_date, format="mixed", yearfirst=True).date().strftime(
                    '%Y-%m-%d') if row.acquisition_date != "None" else None

                df.loc[i] = [filename, row.modality, date, area, area_stdev, density, density_stdev, sanity]

            df.to_csv(os.path.join(output_path, f'{level}_{tag}_{self.database.num_slices*2+1}_slices.csv'), index=False)    

    def stats_to_csv_fast(self, output_path, tag):
        uuids = self.database.uuids
        for level in self.database.v_level:
            ...
        #TODO implement this with polars query



    def image_info_to_csv(self, output_path):
        df = pd.DataFrame(columns=['filename', 'modality', 'acquisition_date', 'spacing_x', 'spacing_y', 'slice_thickness'])
        
        i = 1
        logger.info(f"Writing image resolution to CSV file.")
        #* Iterate over files in input directory
        for i, uuid in enumerate(self.database.uuids):
            with self.database.session as sess:
                row = sess.query(self.database.tables.outputs).filter_by(uuid=uuid).first()
                im_row = sess.query(self.database.tables.images).filter_by(uuid=uuid).first()
            row = dotdict(row._mapping)
            im_row = dotdict(im_row._mapping)
            
            if self.database.file_extension == '.dcm':
                filename = row.pid
            else:
                filename = row.pid + self.database.file_extension

            # date = datetime.strptime(
            #     row.acquisition_date, "%Y%m%d").strftime(
            #         '%Y-%m-%d') if row.acquisition_date != "None" else None
            
            date = pd.to_datetime(
                row.acquisition_date, format="mixed", yearfirst=True).date().strftime(
                '%Y-%m-%d') if row.acquisition_date != "None" else None

            spacing = ast.literal_eval(im_row.spacing)
            df.loc[i] = [filename, row.modality, date, spacing[0], spacing[1], spacing[2]]

        df.to_csv(os.path.join(output_path, f'image_information.csv'), index=False)  


    #~ ==== SANITY PLOTTING =====

    def generate_output_images(self, thresholds):
        tags = list(thresholds.keys())
        tags.append('bone')
        logging.info(f"Plotting masks for these regions: {tags}")
        logger.info(f'Plotting sanity images')
        for uuid in self.database.uuids:
            kwargs = {'uuid': uuid}
            
            #* Query the images and outputs tables with this uuid
            image_row, output_row = self.get_image_info(uuid)
            logger.info(f"Patient id: {image_row.pid} Scan uuid: {uuid}")
            #* Load, reorient (+offset and resample if CBCT)
            Image = self.prepare_image(image_row)

            #* Resample and mip
            #* Resample image and apply gaussian kernel to Gaussian - slow ~2.5s 
            iso_Image, ratio = self.database.resample_isotropic_grid(Image)
            iso_im = sitk.GetArrayFromImage(iso_Image)
            #iso_im = skimage.filters.gaussian(iso_im, sigma=(0, 0, 3), truncate=3.5) #TODO replace this with fft convolution - see observer simulation code.
            mip = np.max(iso_im, axis=-1)

            #* Load and pre-process vertebrae masks
            if output_row.path_to_vertebrae_mask is not None:
                if os.path.isfile(output_row.path_to_vertebrae_mask):
                    logging.info("Reading vertebrae mask")
                    Verts = self.database.load_nifty(output_row.path_to_vertebrae_mask)
                    Verts, _ = self.database.reorient(Verts, orientation='RPI')
                    iso_Verts, ratio = self.database.resample_isotropic_grid(Verts)
                    iso_verts = sitk.GetArrayFromImage(iso_Verts)
                    #iso_im = skimage.filters.gaussian(iso_im, sigma=(0, 0, 3), truncate=3.5) #TODO replace this with fft convolution - see observer simulation code.
                    mip_verts = np.max(iso_verts, axis=-1)
                else:
                    mip_verts = None
            else:
                mip_verts = None

                
            for v_level in self.database.v_level:
                #* Format output directory #
                split = output_row.path_to_bone_masks.split('masks/bone_masks')
                output_dir = os.path.join(split[0], f'sanity/formatted/sanity_{v_level}/{image_row.modality}/')
                os.makedirs(output_dir, exist_ok=True)
                
                logger.info(f"Plotting results at {v_level}")
                im = sitk.GetArrayFromImage(Image)
                if output_row[v_level] is None:
                    logging.error(f"No reference slice at {v_level} for ID: {image_row.pid}")
                    continue
                if output_row[f"{v_level}_path_to_skeletal_muscle_masks"] is None:
                    logging.error(f"{v_level} skeletal muscle mask not detected ID: {image_row.pid}")
                    continue
                
                logging.info(" ----- Only plotting prediction at reference slice ---")
                slice_number = int(output_row[v_level])
                im = im[slice_number]
                if self.database.wm_mode:
                    im -= 1024

                #* Apply window and level to image
                settings = self.database._get_window_level(v_level, image_row.modality)
                im = self.database.wl_norm(im, window=settings['window'], level=settings['level'])

                #*Load prediction(s; if fat segmented)
                predictions = {tag: self.load_predictions(tag, output_row, v_level) for tag in tags}

                params = {
                    'pid': image_row.pid, 
                    'mip': mip, # Sagittal mip
                    'im': im, # Axial slice
                    'ratio': ratio, # Scaling factor due to isotropic resampling
                    'slice_number': slice_number, # Reference slice
                    'v_level': v_level, # Current vertebral level,
                    'mip_verts': mip_verts, # Sagittal mip of vert. mask - used for plotting
                    'image_row': image_row, # Row from images table
                    'output_row': output_row, # Row from outputs table 
                    'predictions': predictions, # Dict of predictions
                    'output_dir': output_dir
                    }
                
                self.plot_predictions(**params)
                 

    def plot_predictions(self, pid, mip, im, ratio, slice_number, 
                        v_level, mip_verts, image_row, output_row, predictions, output_dir):
        #* Plotting
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].axis('off')
        ax[1].axis('off')
        fig.patch.set_facecolor('black')
        #* LEVEL LABELLING
        ax[0].imshow(mip, cmap='gray')
        if mip_verts is not None:
            ax[0].imshow(np.where(mip_verts==0, np.nan, mip_verts), cmap='jet', alpha=0.5)

        for vert in self.database.v_level:
            if output_row[vert] is None:
                continue
            loc = int(output_row[vert])*ratio[0]
            #~ If current level, plot in green
            if vert == v_level:
                ax[0].axhline(loc, c='y', ls='--', linewidth=1.5)
                ax[0].text(0.95, loc+10, vert, c='white')
                ax[0].text(0.95, loc-10, slice_number, c='white')
            else:
                ax[0].axhline(loc, c='r', ls='--', linewidth=1.5, alpha=0.5)
                ax[0].text(0.95, loc+150, vert, c='black', alpha=0.7)

        #* SEGMENTATION
        ax[1].imshow(im, cmap='gray')
        masks = []
        for tag, mask in predictions.items():
            if mask is None:
                logging.info(f"No mask of type {tag} detected - not plotting.")
                continue
            
            masks.append(mask)
                
        # ax[1].imshow(np.where(bone_mask==0, np.nan, 1), alpha=0.95, cmap='Wistia_r')
        masks = np.stack(masks, axis=0)

        #* Concatenate all masks for plotting
        background=np.where(np.sum(masks, axis=0)==0, 1, 0)
        masks = np.concatenate((background[None], masks), axis=0)
        max_mask = np.argmax(masks[:, slice_number], axis=0)
        #* Plot fat masks
        ax[1].imshow(np.where(max_mask==0, np.nan, max_mask), alpha=0.5, cmap='jet')

        if image_row.acquisition_date is None:
            output_name = os.path.join(output_dir, f'{pid}.png')
        else:
            
            date = pd.to_datetime(
                    image_row.acquisition_date, format="mixed", yearfirst=True).date().strftime(
                    '%Y-%m-%d') if image_row.acquisition_date != "None" else None
            
            output_name = os.path.join(
                output_dir, f'{pid}_on_{date}.png') if\
                        date is not None else os.path.join(output_dir, f'{pid}.png')
            
        logger.info(f"Saving image: {output_name}")
        fig.savefig(output_name, bbox_inches='tight', pad_inches=0)
        plt.close()


    def load_predictions(self, tag, row, level):
        if tag != 'bone':
            Prediction = sitk.ReadImage(row[f"{level}_path_to_{tag}_masks"]) if row[f"{level}_path_to_{tag}_masks"] is not None else None
        else: 
            Prediction = sitk.ReadImage(row[f"path_to_{tag}_masks"]) if row[f"path_to_{tag}_masks"] is not None else None

        if Prediction is None:
            return None
        
        return sitk.GetArrayFromImage(Prediction) 
