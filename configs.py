configs = {
    ##############################################
    # Camera Settings
    ##############################################
    'camera_res'      : (1280, 720),   # Camera width & height
                                        # CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT
    'exposure'        : -1,             # CSI camera: -1,0 = auto, 1 max=frame interval
                                        # CAP_PROP_EXPOSURE
    'fps'             : 30,             # CSI: 1/10, 15, 30, 40, 90, 120 overlocked, 180?
    'fourcc'          : -1,             # MJPG, YUY2, for ELP camera https://www.fourcc.org/         CAP_PROP_FOURCC 
                                        # Laptop Windows -1
    'buffersize'      : -1,             # default is 4 for V4L2, max 10,                             CAP_PROP_BUFFERSIZE 
                                        # Laptop: -1
    ##############################################
    # Target Recognition
    ##############################################
    'fov'             : 77,             # camera lens field of view in degress
    ##############################################
    # Target Display
    ##############################################
    'output_res'      : (-1, -1),       # Output resolution 
    'flip'            : 0,              # 0=norotation 
                                        # 1=ccw90deg 
                                        # 2=rotation180 
                                        # 3=cw90 
                                        # 4=horizontal 
                                        # 5=upright diagonal flip 
                                        # 6=vertical 
                                        # 7=uperleft diagonal flip
    'serverfps'       : 16              # frame rate for display server
    }

# 3264 x 2464 FR = 21.0000
# 3264 x 1848 FR = 28.000001
# 1920 x 1080 FR = 29.999999
# 1280 x 720  FR = 120.000005
# 1280 x 720  FR = 59.999999

# Dell integrated camera
# 720p 1280 30
# 540p  960 30
# 360p  640 30
# 480p  640 30
# 240p  320 30
# 180p  320 30
# 120p  160 30

##################################################
# Capture Options for Sony IX219 CSI camera
##################################################
# 4VL2-ctl -L
# v4l2-ctl --list-formats-ext
