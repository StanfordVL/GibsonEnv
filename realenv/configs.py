## Randomize agent initial position & orientation
RANDOM_INITIAL_POSE = True

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
NAVIGATE_MODEL_ID = "sRj553CTHiw"
FETCH_MODEL_ID = "11HB6XZSh1Q"

USE_SENSOR_OUTPUT = True


HIST_MATCHING = False
USE_SEMANTICS = False
SURFACE_NORMAL = False

## Human view camera settings
DEBUG_CAMERA_FOLLOW = True


USE_SMALL_FILLER = False

## Render window settings
HIGH_RES_MONITOR = False
MAKE_VIDEO = False
LIVE_DEMO = False

TASK_POSE = {
    "11HB6XZSh1Q": {
        "navigate": [
            [[0, 0, 3.14], [-2, 3.5, 0.4]],         ## initial
            [[0, 0, 0], [-0.203, -1.74, 1.8]]       ## target
        ],
        "fetch": [
            [[0, 0, 3.14], [-2, 3.5, 0.4]],         ## initial
            [[0, 0, 0], [-0.203, -1.74, 1.8]]       ## target
        ], 
        "climb": [
            #[[0, 0, 3.14], [-2, 3.5, 0.4]],         ## initial
            #[[0, 0, 3.14], [-0.703, -1.24, 2.35]],  ## drop on stairs
            [[0, 0, 3.14], [-2.283, -1.64, 0.65]],  ## bottom of stairs
            #[[0, 0, 0], [-0.203, -1.74, 1.8]]       ## target
            [[0, 0, 3.14/2], [-0.003, -1.54, 1.45]]     ## stairs mid target
        ]
    },
    "sRj553CTHiw": {
        "navigate": [
            [[0, 0, 3 * 3.14/2], [-14.3, 5, 0.5]],  ## initial
            #[[0, 0, 3.14/2], [-14.7, 26.85, 0.5]],
            [[0, 0, 3.14/2], [-4.5607, 40.4859, 0.0991]]
                                                    ## target
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

    }
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




