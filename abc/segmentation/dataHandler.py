"""
Data handling for spine and segmentation
"""
from multiprocessing.sharedctypes import Value
import os
import sqlite3
from telnetlib import OUTMRK
import SimpleITK as sitk
import numpy as np


import sqlalchemy as sa
import ast
import torch
import pandas as pd
from colorama import Fore, Style

from ..mixin import dotdict
import logging

logger = logging.getLogger(__name__)

class segDataSet():
    def __init__(self, database, args, v_level='L3', window=400, level=50,
                  sm_threshold=(-29, +150), fat_threshold=(-190, -30)):
        self.input_dir = args.input_dir
        self.output_dir = args.output_dir

        self.v_level = args.v_level if v_level is None else v_level #* Vertebral level
        self.modality = args.modality #! For filtering data in __getitemm__
        self.path_to_db = args.path_to_db

        self.fat_threshold=fat_threshold
        self.sm_threshold=sm_threshold

        self.db = database
        with self.db.session as sess:
            self.names = [v.pid for v in sess.query(
                self.db.tables.outputs.c.pid).distinct()]    

    def __len__(self):
        # with self.db.session as sess:
        #     rows = sess.query(self.db.tables.images).filter_by(modality=self.modality).all()
        with sqlite3.connect(self.path_to_db, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM images im"
                " INNER JOIN ("
                    f"SELECT o.uuid, o.{self.v_level}, o.modality FROM outputs o"
                    f" WHERE o.{self.v_level} IS NOT NULL"
                    f" AND o.modality='{self.modality}'"
                    ") o on o.uuid = im.uuid"
                " ORDER BY im.pid"
                )
        all_rows = self.cursor_to_df_fetchall(cursor=cursor)
        return len(all_rows.index)
        
    def __getitem__(self, index): 
        """
        Given an index - searches database for all images matching that patient ID
        """
        pid = self.names[index]
        #* Get patient images of correct modality
        with sqlite3.connect(self.path_to_db, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM images im"
                " INNER JOIN ("
                    f"SELECT o.uuid, o.{self.v_level}, o.pid, o.modality FROM outputs o"
                    f" WHERE o.{self.v_level} IS NOT NULL"
                    f" AND o.pid='{pid}'"
                    f" AND o.modality='{self.modality}'"
                    ") o on o.uuid = im.uuid"
                " ORDER BY im.pid"
                )
        all_rows = self.cursor_to_df_fetchall(cursor=cursor)
        if all_rows.empty:
            logger.error(f"No {self.modality}/{self.v_level} detected for patient: {pid}. -- have you added slice numbers?")
            return None
        
        return {'images': all_rows}

    #~-------------------- Utils  ------------------

    @staticmethod
    def cursor_to_df_fetchall(cursor):
        #~ Converts database connection to pd.DataFrame
        desc = cursor.description
        column_names = [col[0] for col in desc]
        dict_ = [dict(zip(column_names, row)) for row in cursor]
        return pd.DataFrame(dict_)

    def query_database(self, table, condition):
        #~ Query a table in the database & filter on PID
        stmt = sa.select(table).where(condition)
        with self.db.session as sess:
            rows = sess.scalars(stmt).first()
        return rows