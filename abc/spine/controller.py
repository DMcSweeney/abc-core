"""
This is the main script in spine pipeline.
Equivalent to model.engine in segmentation pipeline.
"""
import os
import logging
import time
import ast
import json
import shutil
from tqdm import tqdm

from abc_toolkit.stk.mixin import dotdict
from abc_toolkit.stk.spine.app import spineApp
from abc_toolkit.stk.database.database import VERTEBRAL_LEVELS
from abc_toolkit.mixin import dotdict


logger = logging.getLogger(__name__)

class spineController():
    def __init__(self, dataset, args):
        app_dir = os.path.dirname(__file__)
        studies =  "https://127.0.0.1:8989" #TODO How to update and point to XNAT
        config = {
            "models": "find_spine,find_vertebra,segment_vertebra",
            "preload": "false",
            "use_pretrained_model": "true"
        }   

        self.app = spineApp(app_dir, studies, config)

        self.dataset = dataset
        self.db = dataset.db

        self.output_dir = args.output_dir
        logger.info(f"Spine output directory: {self.output_dir}")
        os.makedirs(os.path.join(self.output_dir, 'json'), exist_ok=True)

        self.redo_labelling = args.redo_labelling


    def forward(self):
        # inference and write results
        self.forward_pass()
        # Read results and add to database
        self.update_database()

    def forward_pass(self):
        for data in tqdm(self.dataset):
            if data is None:
                continue
            for _, row in data['images'].iterrows():
                row = dotdict(row)
                logger.info(f"Patient ID: {row.pid} - Scan uuid: {row.uuid}")
            
                mask_output = os.path.join(self.output_dir, 'masks', 'vertebrae', row.pid, row.study_uuid, row.series_uuid,  'mask.nii.gz')
                #mask_output = row.path_to_bone_masks.replace('bone_masks', 'vertebrae')
                json_output = os.path.join(self.output_dir, 'json', row.pid, row.uuid+'.json')
                os.makedirs(os.path.join(self.output_dir, 'json', row.pid), exist_ok=True)

                #* Check if mask and json exist, if yes - skip
                if os.path.isfile(json_output) and not self.redo_labelling:
                    logger.info("Output files detected skipping...")
                    continue

                res = self.app.infer(
                    request={"model": "vertebra_pipeline", "image": row.path_to_im}
                )

                logger.info(f"Inference done for ID: {row.pid} - Scan uuid: {row.uuid}")

                label = res["file"]
                label_json = res["params"]

                if label is None and label_json is None:
                    #* This is the case if no centroids were detected
                    logger.error(f"Inference failed for patient ID: {row.pid} - Scan uuid: {row.uuid}. Could not find centroids.")
                    
                elif label is None and label_json is not None:
                    logger.info(f"Not writing vertebra mask, only saving json with centroids")
                    with open(json_output, 'w') as f:
                        json.dump(label_json, f)

                else:
                    logger.info("Writing segmentations and results json.")
                    shutil.move(label, mask_output)

                    #* Add path to mask to database
                    with self.db.session as sess:
                        cond = {'uuid': row.uuid}
                        #* Find entries by PID and add slice_number
                        sess.query(self.db.tables.outputs).filter_by(
                            **cond).update({'path_to_vertebrae_mask': mask_output})
                        sess.commit()

                    with open(json_output, 'w') as f:
                        json.dump(label_json, f)

    def update_database(self):
        json_dir = os.path.join(self.output_dir, 'json')

        ## Iterate through output json 
        for pid in os.listdir(json_dir):
            pid_path = os.path.join(json_dir, pid)
            for file in os.listdir(pid_path):
                with open(os.path.join(pid_path, file), 'r') as f:
                    data = json.load(f)
                uuid = file.split('.')[0]

                labels, ctrds = data['label_names'], data['centroids']
                        
                #Not needed but negligible time
                vert_lookup = {val: key for key, val in labels.items()}
                ## Organise levels
                slices = self.extract_slice_from_centroid(ctrds, vert_lookup)
                
                #Query db to check if slice needs to be flipped
                with self.db.session as sess:
                    cond = {'uuid': uuid}
                    row = sess.query(self.db.tables.images).filter_by(**cond).first()
                    row = dotdict(row._mapping)

                for level, slice_ in slices.items():
                    if ast.literal_eval(row.flip_axis):
                        slice_num = ast.literal_eval(row.size)[-1] - int(slice_) - 1 #Since size starts at 1 but indexing starts at 0
                        logger.info(f"Patient {row.pid} has been flipped along Z. Slice #: {slice_} -> {slice_num}")
                    else:
                        slice_num = slice_

                    slices[level] = slice_num

                if not slices: #If slices dictionary is empty
                    continue

                logger.info(f"Writing levels to database: {pid}, {slices}")
                ## Write to db
                with self.db.session as sess:
                    cond = {'uuid': uuid}
                    #* Find entries by PID and add slice_number
                    sess.query(self.db.tables.outputs).filter_by(
                        **cond).update(slices)
                    sess.commit()

    @staticmethod
    def extract_slice_from_centroid(centroids, vert_lookup):
        dict_ = {}
        for centroid in centroids:
            for val in centroid.values():
                level = vert_lookup[val[0]]
                if level in VERTEBRAL_LEVELS:
                    dict_[level] = str(val[-1]) 
        return dict_