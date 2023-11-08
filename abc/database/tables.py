"""
SQLite database for application
"""
from sqlalchemy.orm import declarative_base
from sqlalchemy import Table, Column, Integer, String, ForeignKey

Base = declarative_base()


class Tables():
    def __init__(self, v_levels):
        #* Imaging data
        self.images = Table(
            "images",
            Base.metadata,
            Column("id", Integer, primary_key=True),
            Column("pid", String, nullable=False),
            Column("uuid", String, nullable=False),
            Column("path_to_im", String, nullable=False),
            Column("series_uuid", String),
            Column("study_uuid", String),
            Column("file_extension", String),
            Column("modality", String),
            Column("acquisition_date", String),
            Column("spacing", String),
            Column("direction", String),
            Column("size", String),
            Column("origin", String),
            Column("offset", String, default='1'),
            Column("flip_axis", String),
            keep_existing= True
        )
        #* Collecting outputs
        self.outputs = Table(
            "outputs",
            Base.metadata,
            Column("id", Integer, primary_key=True),
            Column("pid", String, nullable=False),
            Column("uuid", String, nullable=False),
            Column("series_uuid", String),
            Column("study_uuid", String),
            Column("modality", String),
            Column("acquisition_date", String),
            
            # Skeletal muscle
            *(Column(f"{val}_skeletal_muscle_area", String) for val in v_levels),
            *(Column(f"{val}_skeletal_muscle_density", String) for val in v_levels),
            *(Column(f"{val}_skeletal_muscle_area_stdev", String) for val in v_levels),
            *(Column(f"{val}_skeletal_muscle_density_stdev", String) for val in v_levels),
            *(Column(f"{val}_path_to_skeletal_muscle_masks", String) for val in v_levels),

            # Subcutaneous fat
            *(Column(f"{val}_subcutaneous_fat_area", String) for val in v_levels),
            *(Column(f"{val}_subcutaneous_fat_density", String) for val in v_levels),
            *(Column(f"{val}_subcutaneous_fat_area_stdev", String) for val in v_levels),
            *(Column(f"{val}_subcutaneous_fat_density_stdev", String) for val in v_levels),
            *(Column(f"{val}_path_to_subcutaneous_fat_masks", String) for val in v_levels),
            
            # Visceral fat
            *(Column(f"{val}_visceral_fat_area", String) for val in v_levels),
            *(Column(f"{val}_visceral_fat_density", String) for val in v_levels),
            *(Column(f"{val}_visceral_fat_area_stdev", String) for val in v_levels),
            *(Column(f"{val}_visceral_fat_density_stdev", String) for val in v_levels),
            *(Column(f"{val}_path_to_visceral_fat_masks", String) for val in v_levels),

            # IMAT
            *(Column(f"{val}_IMAT_area", String) for val in v_levels), # Intramuscular adipose tissue
            *(Column(f"{val}_IMAT_density", String) for val in v_levels),
            *(Column(f"{val}_IMAT_area_stdev", String) for val in v_levels), 
            *(Column(f"{val}_IMAT_density_stdev", String) for val in v_levels),
            *(Column(f"{val}_path_to_IMAT_masks", String) for val in v_levels),

            # Body area
            *(Column(f"{val}_body_area", String) for val in v_levels), # IBody mask
            *(Column(f"{val}_body_density", String) for val in v_levels),
            *(Column(f"{val}_body_area_stdev", String) for val in v_levels), 
            *(Column(f"{val}_body_density_stdev", String) for val in v_levels),
            *(Column(f"{val}_path_to_body_masks", String) for val in v_levels),

            Column("path_to_bone_masks", String),
            Column("path_to_vertebrae_mask", String),
            *(Column(val, String) for val in v_levels),
            *(Column(f'{val}_sanity_check', String) for val in v_levels),
            keep_existing= True
        )

        #* Table used for undo-ing pre-processing of labelling inputs
        self.interim = Table(
            "interim",
            Base.metadata,
            Column("id", Integer, primary_key=True),
            Column("pid", String, nullable=False),
            Column("uuid", String, nullable=False),
            Column("modality", String),
            Column("spacing", String),
            Column("direction", String),
            Column("size", String),
            Column("origin", String),
            keep_existing= True
        )