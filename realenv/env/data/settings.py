"""
  Name: settings.py
  Author: Sasha Sax, CVGL
  Desc: Contains all the settings that our scripts will use.

  Usage: for import only
"""
from math import pi
import math
import sys


# Images
CREATE_FIXATED = False
CREATE_NONFIXATED = False
CREATE_PANOS = True
USE_ONLY_ENABLED_CAMERAS = False
POINT_TYPE = 'SWEEP' # 'CORRESPONDENCES' # The basis for how points are generated
#POINT_TYPE = 'CORRESPONDENCES' # The basis for how points are generated

# File paths
BLENDER_PATH = '/Applications/Blender/blender.app/Contents/MacOS/blender'
CAMERA_IS_ENABLED_FILE = "ActiveSweeps.json"
CAMERA_POSE_FILE = "sweep_locations.csv"
FILE_NAMING_CONVENTION = 'DEBUG'  # Or 'STANDARD'
LOG_FILE = sys.stderr  # Use sys.stderr to avoid Blender garbage
LEGO_MODEL_FILE = "out_res.obj" # "out_res.obj" 
SEMANTIC_MODEL_FILE = "semantic_lookup.obj" # "out_res.obj" 
SEMANTIC_PRETTY_MODEL_FILE = "semantic.obj" # "out_res.obj" 
MODEL_FILE = "out_res.ply" # "out_res.obj" 
PANO_VIEW_NAME = 'equirectangular'
PREFERRED_IMG_EXT = 'PNG' # PNG, JPEG
POINTS_DIR = "points"

# Render settings
RESOLUTION = 1080
SENSOR_DIM = 20  # 20
STOP_VIEW_NUMBER = -1  #2 # Generate up to (and including) this many views. -1 to disable.
DEBUG_AT_POINT = None
DEBUG_AT_VIEW = None
TILE_SIZE = 128
PANO_RESOLUTION = (2048, 1024)

# Color depth
COLOR_BITS_PER_CHANNEL = '8'  # bits per channel. PNG allows 8, 16.
DEPTH_BITS_PER_CHANNEL = '16'  # bits per channel. PNG allows 8, 16.
DEPTH_MAX_DISTANCE_METERS = 128.  # With 128m and 16-bit channel, has sensitivity 1/512m (128 / 2^16)
MIST_MAX_DISTANCE_METERS = 128.  # With 128m and 16-bit channel, has sensitivity 1/512m (128 / 2^16)

# Field of view a
BLUR_ANGLE_FROM_PLANE_OF_ROTATION = math.radians( 60 ) # 60d, use pi/2 for testing pi / 2. #  
FIELD_OF_VIEW_RADS = math.radians( 60 ) 
FIELD_OF_VIEW_MIN_RADS = math.radians( 45 )
FIELD_OF_VIEW_MAX_RADS = math.radians( 75 ) 
FIELD_OF_VIEW_MATTERPORT_RADS = math.radians( 90 )
LINE_OF_SITE_HIT_TOLERANCE = 0.001  # Matterport has 1 unit = 1 meter, so 0.001 is 1mm
MODE = 'DEBUG' # DEBUG, TEST, PRODUCTION


# Debugging
VERBOSITY_LEVELS = { 'ERROR': 0,  # Everything >= VERBOSITY will be printed 
                     'WARNING': 20,
                     'STANDARD': 50, 
                     'INFO': 90,
                     'DEBUG': 100  } 
VERBOSTITY_LEVEL_TO_NAME = { v: k for k, v in VERBOSITY_LEVELS.items()}
VERBOSITY = VERBOSITY_LEVELS[ 'INFO' ]
RANDOM_SEED = 42  # None to disable

# TEST_SETTINGS
NORMALS_SIG_FIGS_TOLERANCE = 4
FLOAT_PLACES_TOLERANCE = 4

# DO NOT CHANGE -- effectively hardcoded
CYCLES_DEVICE = 'GPU'         # Not yet implemented!
EULER_ROTATION_ORDER = 'XYZ'  # Not yet implemented!
MATTERPORT_SKYBOX_RESOLUTION = 1024

# NOT YET IMPLEMENTED -- changing it won't make a difference
MAX_PITCH_SIGMA = 3 # how many sigma = the max pitch


# AUTOMATICALLY CALCULATED SETTINGS:
MAX_ANGLE_OF_CAMERA_BASE_FROM_PLANE_OF_ROTATION = float( BLUR_ANGLE_FROM_PLANE_OF_ROTATION ) - ( FIELD_OF_VIEW_MAX_RADS / 2.) #pi / 12 TODO: Set this back when not testing
