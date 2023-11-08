"""
Script holding methods for launching app in different ways
"""
from gooey import Gooey, GooeyParser
from configparser import ConfigParser
import json
import os
from colorama import Fore

@Gooey(program_name='SarcopeniaTool', 
    poll_external_updates=False, 
    default_size=(800, 1000),
    image_dir='./icons/')
def gui_launch():
    #~ Query and load arguments
    #TODO handle supplementary data
    parser = GooeyParser(description="SarcopeniaTool v.0.2")
    parser.add_argument('input_dir', metavar='Input Directory', widget='DirChooser', 
    gooey_options={
        'initial_value': '/home/donal/SarcopeniaTool/data'})

    parser.add_argument('output_dir', metavar='Output Directory', widget='DirChooser', 
    gooey_options={
        'initial_value': '/home/donal/spineTool/outputs'})

    parser.add_argument('level_labelling', metavar='Detect specified vertebral level', 
    widget='CheckBox', gooey_options={'initial_value': True})

    parser.add_argument('segmentation', metavar='Segment skeletal muscle at specified level', 
    widget='CheckBox', gooey_options={'initial_value': True})

    parser.add_argument('v_level', metavar='Vertebral Level', widget='Dropdown', 
    choices=['C3', 'T12', 'L3', 'Thigh'], gooey_options={'initial_value': 'C3'})

    parser.add_argument('modality', metavar='Imaging modality', widget='Dropdown', 
    choices=['PCT', 'CBCT'], gooey_options={'initial_value': 'PCT'})

    parser.add_argument('--wm_mode', metavar='WorldMatch mode?', 
        help='Account for offset from WM pre-processing', action="store_true",
        gooey_options={'initial_value': True})

    parser.add_argument('--normalise', metavar='Apply Window/Level?', 
        action="store_true", gooey_options={'initial_value': True})

    parser.add_argument('--sanity', metavar='Save sanity images', 
        action="store_true", gooey_options={'initial_value': True})
    parser.add_argument('--verbose', metavar='Enable print statements', 
        action="store_true", gooey_options={'initial_value': False})

    parser.add_argument('--slice_numbers', metavar='CSV of slice numbers', 
    widget='FileChooser', gooey_options={'initial_value': None})

    return parser.parse_args()

def config_launch(config_file):
    """
    Launch from config file
    """
    
    #assert config_file in os.listdir('.'), Fore.RED + 'Config file not found in ./'
    
    config = ConfigParser()
    config.read(config_file) 

    class argParse():
        def __init__(self, config):
            self.input_dir = config['DIRECTORIES']['input_dir']
            self.output_dir = config['DIRECTORIES']['output_dir']

            self.level_labelling = config['RUNTIME'].getboolean('levelLabelling')
            self.segmentation = config['RUNTIME'].getboolean('segmentation')
            self.extract_stats = config['RUNTIME'].getboolean('extract_stats')
            
            v_level = config["DEFAULTS"]["v_level"] 
            if ',' in v_level:
                self.v_level = v_level.split(',')
            else:
                self.v_level = [v_level]
            
            self.modality = config['DEFAULTS']['modality']
            self.wm_mode = config['DEFAULTS'].getboolean('wm_mode')
            self.normalise = config['DEFAULTS'].getboolean('normalise')
            self.sanity = config['DEFAULTS'].getboolean('sanity')
            self.verbose = config['DEFAULTS'].getboolean('verbose')
            
            self.ext_inputs = config['SUPPLEMENTARY_INPUTS']['directory']
            self.ext_slice_numbers = config['SUPPLEMENTARY_INPUTS']['slice_numbers']
            self.ext_offset = config['SUPPLEMENTARY_INPUTS']['cbct_offset']
            self.restrict_to_csv = config['SUPPLEMENTARY_INPUTS'].getboolean('restrict_to_csv')

    args = argParse(config)
    return args