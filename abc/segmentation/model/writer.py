"""
Plotting methods
"""
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import matplotlib
import os
import SimpleITK as sitk

matplotlib.use('agg')
from ...mixin import dotdict

class segWriter():
    def __init__(self, engine, modality, dataset):
        self.modality = modality #! Useful for plotting?
        self.dataset = dataset
        self.engine = engine
        self.db = engine.db
        self.output_dir = engine.output_dir
        self.v_level = engine.v_level

    def save_sanity(self):
        self.plot_predictions()
    
    @staticmethod
    def wl_norm(img, window, level):
        minval = level - window/2
        maxval = level + window/2
        wld = np.clip(img, minval, maxval)
        wld -= minval
        wld /= window
        return wld

    @staticmethod
    def plot_img(img, output_name):
        img = np.squeeze(img)[0]
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.axis('off')
        ax.imshow(img, cmap='gray')
        fig.savefig(output_name)
        plt.close()

    @staticmethod
    def resample_cbct2pct(pct, cbct):
        return sitk.Resample(cbct, pct, sitk.Transform(), sitk.sitkLinear, 0, pct.GetPixelID())

    @staticmethod
    def reorient(Image, orientation='RPI'):
        orient = sitk.DICOMOrientImageFilter()
        orient.SetDesiredCoordinateOrientation(orientation)
        return orient.Execute(Image)

    def plot_predictions(self):
        #~ Save sanity check (labelling) image
        with self.db.session as sess:
            names = [v.pid for v in sess.query(self.db.tables.outputs.c.pid).distinct()]

        for name in names:
            cond = {'pid': name,'modality': self.modality}
            with self.db.session as sess:
                rows = sess.query(self.db.tables.outputs).filter_by(**cond).all()
            if rows:
                for i, row in enumerate(rows):
                    row = dotdict(row)
                    uuid = row.uuid
                    pid = row.pid
                    with self.db.session as sess:
                        kwargs = {'uuid': uuid}
                        im_row = sess.query(self.db.tables.images).filter_by(
                            **kwargs).first()
                        out_row = sess.query(self.db.tables.outputs).filter_by(
                            **kwargs).first()
                        im_row = dotdict(im_row)
                        out_row = dotdict(out_row)
                    #* Load image            
                    Image = self.db.loader_function(im_row.path_to_im)
                    Image = self.reorient(Image, orientation='RPI')
                    Image = sitk.Cast(Image, sitk.sitkFloat32)
                    Image = Image*float(im_row.offset)
                    if im_row.modality == 'CBCT':
                        with self.db.session as sess:
                            conditions = {'pid': im_row.pid, 'modality': 'CT'}
                            pct_row = sess.query(self.db.tables.images).filter_by(**conditions).first()
                        pct = self.db.loader_function(pct_row.path_to_im)
                        pct = self.reorient(pct, orientation='RPI')
                        Image = self.resample_cbct2pct(pct, Image)

                    im = sitk.GetArrayFromImage(Image)
                    slice_number = int(out_row[self.v_level])
                    im = im[slice_number]

                    if self.dataset.wm_mode:
                        im -= 1024
                    if self.dataset.normalise:
                        im = self.wl_norm(im, window=self.dataset.settings['window'], level=self.dataset.settings['level'])

                    #*Load prediction
                    with np.load(out_row[f"{self.v_level}_path_to_masks"]) as f:
                        mask = f['mask']
                        
                    with np.load(out_row.path_to_bone_masks) as f:
                        bone_mask = f['mask'][slice_number]

                    #* Format output directory 
                    output_dir = out_row.path_to_bone_masks.replace('masks/bone_masks', f'sanity/skeletal_muscle_segmentation/sanity_{self.v_level}')
                    output_dir = output_dir.rstrip('/mask.npz')
                    os.makedirs(output_dir, exist_ok=True)
                    
                    #* Plotting
                    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                    ax.axis('off')
                    ax.imshow(im, cmap='gray')
                    ax.imshow(np.where(bone_mask==0, np.nan, 2), alpha=0.4, cmap='Set1')
                    ax.imshow(np.where(mask==0, np.nan, 1), alpha=0.4, cmap='viridis')
                    fig.savefig(os.path.join(output_dir, f'{pid}.png'))
                    plt.close()