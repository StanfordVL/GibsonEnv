import numpy as np
from enum import Enum


TEST_MODE = False

## Randomize agent initial position & orientation
RANDOM_INITIAL_POSE = False
RANDOM_TARGET_POSE  = False

RANDOM_RANGE = "TINY"

RANDOM_TARGET_POSE_TEST  = False


RANDOM_RANGE = "LARGE"

LEARNING_RATE_NAME = "LARGE"

if RANDOM_RANGE == "TINY":
    RANDOM_TARGET_RANGE = 0.025
    RANDOM_INITIAL_RANGE_X = [-0.1, 0.05]
    RANDOM_INITIAL_RANGE_Y = [-0.025, 0.025]
    RANDOM_INITIAL_RANGE_Z = [0, 0]
    RANDOM_INITIAL_RANGE_DEG = [-np.pi/18, np.pi/18]
elif RANDOM_RANGE == "SMALL":
    RANDOM_TARGET_RANGE = 0.05
    RANDOM_INITIAL_RANGE_X = [-0.1, 0.05]
    RANDOM_INITIAL_RANGE_Y = [-0.05, 0.05]
    RANDOM_INITIAL_RANGE_Z = [0, 0.2]
    RANDOM_INITIAL_RANGE_DEG = [-np.pi/9, np.pi/9]
elif RANDOM_RANGE == "MID":
    RANDOM_TARGET_RANGE = 0.2
    RANDOM_INITIAL_RANGE_X = [-0.2, 0.1]
    RANDOM_INITIAL_RANGE_Y = [-0.1, 0.1]
    RANDOM_INITIAL_RANGE_Z = [0, 0.2]
    RANDOM_INITIAL_RANGE_DEG = [-np.pi/4, np.pi/4]
elif RANDOM_RANGE == "LARGE":
    RANDOM_TARGET_RANGE = 0.5
    RANDOM_INITIAL_RANGE_X = [-0.4, 0.1]
    RANDOM_INITIAL_RANGE_Y = [-0.1, 0.1]
    RANDOM_INITIAL_RANGE_Z = [0, 0.4]
    RANDOM_INITIAL_RANGE_DEG = [-np.pi/2, np.pi/2]
else:
    RANDOM_TARGET_RANGE = 1
    RANDOM_INITIAL_RANGE_X = [-0.6, 0.1]
    RANDOM_INITIAL_RANGE_Y = [-0.1, 0.1]
    RANDOM_INITIAL_RANGE_Z = [0, 0.4]
    RANDOM_INITIAL_RANGE_DEG = [-2 *np.pi/3, 2* np.pi/3]

if LEARNING_RATE_NAME == "SMALL":
    LEARNING_RATE = 3e-5
elif LEARNING_RATE_NAME == "MID":
    LEARNING_RATE = 3e-4
elif LEARNING_RATE_NAME == "LARGE":
    LEARNING_RATE = 3E-6


CHOOSE_SMALL_RANDOM_RANGE = False
CHOOSE_TINY_RANDOM_RANGE = False


CAPTURE_DIR = "/home/jerry/Dropbox/Cambria Paper/Experiments"

ENABLE_PROFILING = True

## WORKAROUND (hzyjerry): scaling building instead of agent, this is because
## pybullet doesn't yet support downscaling of MJCF objects
MJCF_SCALING  = 0.6
USE_MJCF = True

## Small model: 11HB6XZSh1Q
## Psych model: BbxejD15Etk
## Gates 1st: sRj553CTHiw
## Basement: 13wHkWg1BWZ
## Street scene: 15N3xPvXqFR
## Gates 3rd: TVHnHa4MZwE

CLIMB_MODEL_ID = "sRj553CTHiw"
NAVIGATE_MODEL_ID = "sRj553CTHiw"
FETCH_MODEL_ID = "11HB6XZSh1Q"

if TEST_MODE:
    CLIMB_MODEL_ID = "test"
    NAVIGATE_MODEL_ID = "test"
    FETCH_MODEL_ID = "test"


USE_SENSOR_OUTPUT = True
USE_GPS_OUTPUT = False

HIST_MATCHING = False
USE_SMALL_FILLER = False
USE_SMOOTH_MESH = False

DISABLE_FILLER = False

## Human view camera settings
DEBUG_CAMERA_FOLLOW = True


DISPLAY_UI = True
ENABLE_UI_RECORDING = False
class UIMode(Enum):
    UI_ONE = 1
    UI_TWO = 2
    UI_THREE = 3
    UI_FOUR = 4
    UI_FIVE = 5
    UI_SIX = 6
    UI_NONE = 0
    def is_valid(mode):
        return mode in [UIMode.UI_ONE, UIMode.UI_TWO, UIMode.UI_THREE, UIMode.UI_FOUR, UIMode.UI_FIVE, UIMode.UI_SIX, UIMode.UI_NONE]

UI_MODE = UIMode.UI_TWO

class View(Enum):
    EMPTY = 0
    RGB_FILLED = 1
    RGB_PREFILLED = 2
    DEPTH = 3
    NORMAL = 4
    SEMANTICS = 5
    PHYSICS = 6

class ViewComponent:
    ONE   = [View.RGB_FILLED]
    TWO   = [View.RGB_FILLED, View.DEPTH]
    FOUR  = [View.RGB_FILLED, View.DEPTH, View.NORMAL, View.SEMANTICS]
    THREE = [View.RGB_FILLED, View.DEPTH, View.NORMAL]
    def getComponents():
        assert(UIMode.is_valid(UI_MODE))
        if UI_MODE == UIMode.UI_ONE:
            return ViewComponent.ONE
        if UI_MODE == UIMode.UI_TWO:
            return ViewComponent.TWO
        if UI_MODE == UIMode.UI_THREE:
            return ViewComponent.THREE
        if UI_MODE == UIMode.UI_FOUR:
            return ViewComponent.FOUR
        if UI_MODE == UIMode.UI_FIVE:
            return ViewComponent.FIVE
        if UI_MODE == UIMode.UI_SIX:
            return ViewComponent.SIX
        if UI_MODE == UIMode.UI_NONE:
            return None

## Render window settings
HIGH_RES_MONITOR = False
MAKE_VIDEO = True
LIVE_DEMO = False
ENABLE_UI_RECORDING = False

PUT_RED_FLAGS = True

TASK_POSE = {

    "Gn4zCib6aCV": {
        "navigate": [
            [[0, 0, 3.14/2], [7, -7, 1.2]],
            [[0, 0, 0], [0, 0, 0]]
        ],
        "fetch": [
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]]
        ],
        "climb": [
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]]
        ]
    },  # basement house


    "8ohVSvgSpxP": {
        "navigate": [
            [[0, 0, -3.14/2], [1, 1, 1.2]],
            [[0, 0, 0], [0, 0, 0]]
        ],
        "fetch": [
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]]
        ],
        "climb": [
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]]
        ]
    },  # basement house

    "4sJvhG2KCVH": {
        "navigate": [
            [[0, 0, 3.14/2], [-10, 10, 2]],
            [[0, 0, 0], [0, 0, 0]]
        ],
        "fetch": [
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]]
        ],
        "climb": [
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]]
        ]
    },  # basement house


    "gJTEGVwrRVM": {
        "navigate": [
            # [[0, 0, 3.14/2], [-2, 3.5, 1.1]],       ## for making visual
            # [[0, 0, 3.14], [-2, 3.5, 0.4]],         ## initial
            [[0, 0, 3.14/2], [-0.8, -1.9, 1.4]]  ## target
        ],
        "fetch": [
            [[0, 0, 3.14], [-2, 3.5, 0.4]],  ## initial
            [[0, 0, 0], [-0.203, -1.74, 1.8]]  ## target
        ],
        "climb": [
            # [[0, 0, 3.14], [-0.703, -1.24, 2.35]],         ## drop on stairs
            # [[0, 0, 0], [-0.203, -1.74, 1.8]],             ## Debug: slightly above stairs
            # [[0, 3.14 / 2, 3.14], [-2.283, -1.64, 0.65]],  ## Debug: leaning too much
            # [[0, 0, 3.14], [-2.283, -1.64, 0.15]],         ## bottom of stairs
            # [[3.14/4, 0, 3.14], [-1.883, -1.64, 0.75]],    ## bottom of stairs, a bit up
            [[0, 0, 0], [-0.003, -1.64, 1.65]],  ## starting at stairs target
            # [[0, 3.14/4, 3.14], [-1.403, -1.64, 0.95]],    ## starting at stairs 1/4


            # [[0, 0, 3.14], [-2.283, -0.64, 0.15]],  ## target bottom of stairs, closer to living room
            # [[0, 0, 3.14], [2 * -2.583, -1.64, 0.15]],   ## zoomed target at bottom of stairs
            [[0, 0, 3.14], [2.583, -1.64, 0.15]],  ## target at bottom of stairs

            # [[0, 0, 3.14/2], [-1.403, -1.84, 1.75]],
            # [[0, 0, 3.14], [-2, 3.5, 0.15]]         ## target living room
            # [[0, 0, 3.14/2], [-0.003, -1.84, 1.45]] ## target stairs target
            # [[0, 0, 3.14/2], [-1.403, -1.84, 0.75]] ## target stairs half way
        ]
    },

    "11HB6XZSh1Q": {
        "navigate": [
            #[[0, 0, 3.14/2], [-2, 3.5, 1.1]],       ## for making visual
            #[[0, 0, 3.14], [-2, 3.5, 0.4]],         ## initial
            [[0, 0, 3.14], [-0.8, -1.9, 2.8]]       ## target
        ],
        "fetch": [
            [[0, 0, 3.14], [-2, 3.5, 0.4]],         ## initial
            [[0, 0, 0], [-0.203, -1.74, 1.8]]       ## target
        ],
        "climb": [
            #[[0, 0, 3.14], [-0.703, -1.24, 2.35]],         ## drop on stairs
            #[[0, 0, 0], [-0.203, -1.74, 1.8]],             ## Debug: slightly above stairs
            #[[0, 3.14 / 2, 3.14], [-2.283, -1.64, 0.65]],  ## Debug: leaning too much
            #[[0, 0, 3.14], [-2.283, -1.64, 0.15]],         ## bottom of stairs
            #[[3.14/4, 0, 3.14], [-1.883, -1.64, 0.75]],    ## bottom of stairs, a bit up
            [[0, 0, 0], [-0.303, -1.64, 1.45]],             ## starting at stairs target
            #[[0, 3.14/4, 3.14], [-1.403, -1.64, 0.95]],    ## starting at stairs 1/4


            #[[0, 0, 3.14], [-2.283, -0.64, 0.15]],  ## target bottom of stairs, closer to living room
            #[[0, 0, 3.14], [2 * -2.583, -1.64, 0.15]],   ## zoomed target at bottom of stairs
            [[0, 0, 3.14], [2.583, -1.64, 0.15]],   ## target at bottom of stairs

            #[[0, 0, 3.14/2], [-1.403, -1.84, 1.75]],
            #[[0, 0, 3.14], [-2, 3.5, 0.15]]         ## target living room
            #[[0, 0, 3.14/2], [-0.003, -1.84, 1.45]] ## target stairs target
            #[[0, 0, 3.14/2], [-1.403, -1.84, 0.75]] ## target stairs half way
        ]
    },
    "sRj553CTHiw": {
        "navigate": [
            #[[0, 0, 3.14/2], [-14.0747, 17.5126, 0.5]],
            [[0, 0, 3 * 3.14/2], [-14.3, 5, 1.2]],  ## initial: end of hall way
            #[[0, 0, 3.14/2], [-14.7, 26.85, 0.5]],
            #[[0, 0, 3.14/2], [-4.5607, 40.4859, 0.0991]] ## target: gates entrance hall
            #[[0, 0, 0], [-8.6773, 1.4495, 1.2]]
            #[[0, 0, 3.14/2], [-14.0747, 17.5126, 1.5]], ## for minitaur
            #[[0, 0, 3.14/2], [-14.0747, 17.5126, 0.5]],
            [[0, 0, 3 * 3.14/2], [-14.3, 5, 0.5]],  ## initial: end of hall way
            #[[0, 0, 3.14/2], [-14.7, 26.85, 0.5]], ## near silvio's room
            [[0, 0, 3.14/2], [-14.3, 45.07, 0.5]],  ## down silvio's room
            #[[0, 0, 3.14/2], [-4.5607, 40.4859, 0.0991]] ## target: gates entrance hall
            #[[0, 0, 0], [-8.6773, 1.4495, 0.5]]

        ],
        "fetch": [
            [[0, 0, 3 * 3.14/2], [-14.3, 5, 0.5]],  ## initial
            [[0, 0, 3.14], [-7, 2.6, 0.5]]          ## target
        ],
        "climb": [
            [[0, 0, 3.14], [-7, 2.6, 0.5]],
            [[0, 0, 3.14], [-7, 2.6, 0.5]],
        ]
    },
    "BbxejD15Etk": {
        "navigate": [
            [[0, 0, 0], [0, 1, 1.2]],
            [[0, 0, 0], [0, 0, 0]]
        ],
        "fetch": [
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]]
        ],
        "climb": [
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]]
        ]
    },
    "13wHkWg1BWZ": {
        "navigate": [
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]]
        ],
        "fetch": [
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]]
        ],
        "climb": [
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]]
        ]
    },  # basement house
    "TVHnHa4MZwE": {
        "navigate": [
            [[0, 0, 2.5 * 3.14/4] , [7.7417, 5.1469, 0.9726]],
            [[0, 0, 3.14], [12.5945, -4.8648, 1.5727]],
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]]
        ],
        "fetch": [
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]]
        ],
        "climb": [
            #[[0, 0, 3.14/2], [11.5945, -1.8648, 0.8727]],
            [[0, 0, 3.14], [12.5945, -4.8648, 0.5727]],


            [[0, 0, 0], [16.06, -5.07, -0.927]],  ## end of first half
            #[[0, 0, 0], [15.0, -1.92, -1.65]],    ## end of second half, right
            #[[0, 0, 0], [15.60, -7.98, -1.64]]    ## end of second half, left
        ]
    },
    "test": {
        "navigate": [
            [[0, 0, 3 * 3.14/2], [-14.3, 5, 1.2]],  ## initial: end of hall way
            [[0, 0, 3 * 3.14/2], [-14.3, 5, 0.5]],  ## initial: end of hall way
            [[0, 0, 3.14/2], [-14.3, 45.07, 0.5]],  ## down silvio's room
        ],
        "fetch": [
            [[0, 0, 3 * 3.14/2], [-14.3, 5, 0.5]],  ## initial
            [[0, 0, 3.14], [-7, 2.6, 0.5]]          ## target
        ],
        "climb": [
            [[0, 0, 3.14], [-7, 2.6, 0.5]],
            [[0, 0, 3.14], [-7, 2.6, 0.5]],
        ]
    },
}

OFFSET_GROUND = {
    "TVHnHa4MZwE": -3,
    "BbxejD15Etk" : -2,
    "8ohVSvgSpxP" : -2
}

## Initial locations
INITIAL_POSE = {
    "humanoid": {
        "11HB6XZSh1Q": [
            [[0, 0, 3.14/2], [-4.655, -9.038, 1.532]],
            [[0, 0, 3 * 3.14/2], [-3.38, -7, 1.4]],         ## living room open area
            [[0, 0, 3 * 3.14/2], [-4.8, -5.2, 1.9]],        ## living room kitchen table
            [[0, 0, 3.14], [-0.603, -1.24, 2.35]],          ## stairs
        ],
        "BbxejD15Etk": [                                 ## Psych building
            [[0, 0, 3 * 3.14/2], [-6.76, -12, 1.4]]
        ],
        "15N3xPvXqFR": [
            [[0, 0, 3 * 3.14/2], [-0, -0, 1.4]]
        ],
        "TVHnHa4MZwE":[
            [[0, 0, 0], [0, 0, 0]]
        ]
    },
    "husky": {
        "11HB6XZSh1Q": [
            [[0, 0, 3.14], [-2, 3.5, 0.4]],  ## living room
            [[0, 0, 0], [-0.203, -1.74, 1.8]]  ## stairs
        ],
        "sRj553CTHiw": [
            [[0, 0, 3 * 3.14/2], [-14.3, 5, 0.5]],
            [[0, 0, 3.14], [-7, 2.6, 0.5]],
            [[0, 0, 3.14/2], [-13.2, 39.7, 0.5]],
            [[0, 0, 3.14/2], [-14.7, 26.85, 0.5]],
            [[0, 0, 3.14/2], [-4.5607, 40.4859, 0.0991]]
        ],
        "BbxejD15Etk": [
            [[0, 0, 3.14], [0, 0, 0.4]],
        ],
        "13wHkWg1BWZ": [  # basement house
            [[0, 0, 3.14], [-1, -1, -0.4]],
        ]
    },
    "quadruped": {
        "11HB6XZSh1Q": [
            [[0, 0, 3.14], [-2, 3.5, 0.4]],  ## living room
            [[0, 0, 0], [-0.203, -1.74, 1.8]]  ## stairs
        ]
    },
    "ant": {
        "11HB6XZSh1Q": [
            [[0, 0, 3.14], [-2.5, 5.5, 0.4]] ## living room couch
        ]
    }
}




