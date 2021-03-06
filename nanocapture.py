###############################################################################
# CSi video capture on Jetson Nano using gstreamer
# Unfortunately latency is expected to be 100-200ms
# Urs Utzinger
# 2020
###############################################################################

###############################################################################
# Imports
###############################################################################

# Multi Threading
from threading import Thread
from threading import Lock

#
import logging
import time
import os

# Open Computer Vision
import cv2

# Camera configuration file
from configs   import configs

###############################################################################
# Video Capture
###############################################################################

def gstreamer_pipeline(
        capture_width=1920, capture_height=1080,
        display_width=1280, display_height=720,
        framerate=30, exposure_time=-1, # ms
        flip_method=0):

    #        'timeout=0 '                                # 0 - 2147483647
    #        'blocksize=-1 '                             # block size in bytes
    #        'num-buffers=-1 '                           # -1..2147483647 (-1=ulimited) 
    #                                                    # num buf before sending EOS
    #        'sensor-mode=-1 '                           # -1..255, IX279 
    #                                                    # 0(3264x2464,21fps)
    #                                                    # 1 (3264x1848,28)
    #                                                    # 2(1080p.30),
    #                                                    # 3(720p,60),
    #                                                    # 4(=720p,120)
    #        'tnr-strength=-1 '                          # -1..1
    #        'tnr-mode=1 '                               # 0,1,2
    #        # edge enhancement does not accept settings
    #        #'ee-mode=0'                               # 0,1,2
    #        #'ee-strength=-1 '                         # -1..1
    #        'aeantibanding=1 '                         # 0..3, off,auto,50,60Hz
    #        'bufapi-version=false '                    # new buffer api
    #        'maxperf=true '                            # max performance
    #        'silent=true '                             # verbose output
    #        'saturation=1 '                            # 0..2
    #        'wbmode=1 '                                # white balance mode, 0..9 0=off 1=auto
    #        'awblock=false '                           # auto white balance lock
    #        'aelock=true '                             # auto exposure lock
    #        'exposurecompensation=0 '                  # -2..2
    #        'exposuretimerange='                       # 
    #        'gainrange="1.0 10.625" '                  # "1.0 10.625"
    #        'ispdigitalgainrange="1 8" '               

    if exposure_time <= 0:
        # auto exposure
        ################
        nvarguscamerasrc_str = (
            'nvarguscamerasrc '        +
            'do-timestamp=true '       +
            'maxperf=false '            +
            'silent=true '             +
            'awblock=false '           +
            'aelock=false '            +
            'exposurecompensation=0 ')
    else:
        # static exposure
        #################
        exposure_time = exposure_time * 1000000 # ms to ns
        exp_time_str = '"' + str(exposure_time) + ' ' + str(exposure_time) + '" '
        nvarguscamerasrc_str = (
            'nvarguscamerasrc '          +
            'do-timestamp=true '         +
            'timeout=0 '                 +
            'blocksize=-1 '              +
            'num-buffers=-1 '            +
            'sensor-mode=-1 '            +
            'tnr-strength=-1 '           +
            'tnr-mode=1 '                +
            'aeantibanding=1 '           +
            'bufapi-version=false '      +
            'maxperf=true '              +
            'silent=true '               +
            'saturation=1 '              +
            'wbmode=1 '                  +
            'awblock=false '             +
            'aelock=true '               +
            'exposurecompensation=0 '    +
            'exposuretimerange='         +
            exp_time_str)

    # Flip options
    #   0=norotation
    #   1=ccw90deg
    #   2=rotation180
    #   3=cw90
    #   4=horizontal
    #   5=uprightdiagonal flip
    #   6=vertical
    #   7=uperleft flip

    gstreamer_str = (
        '! video/x-raw(memory:NVMM), '                       +
        'width=(int){:d}, '.format(capture_width)            +
        'height=(int){:d}, '.format(capture_height)          +
        'format=(string)NV12, '                              +
        'framerate=(fraction){:d}/1 '.format(framerate)      +
        '! nvvidconv flip-method={:d} '.format(flip_method)  +
        '! video/x-raw, width=(int){:d}, height=(int){:d}, format=(string)BGRx '.format(display_width,display_height) +
        '! videoconvert '                                    +
        '! video/x-raw, format=(string)BGR '                 +
        '! appsink')

    return (
        nvarguscamerasrc_str + gstreamer_str
    )

class nanoCapture(Thread):
    """
    This thread continually captures frames from a CSI camera
    """

    # Initialize the Camera Thread
    # Opens Capture Device and Sets Capture Properties
    def __init__(self, camera_num: int = 0, res: (int, int) = None, 
                 exposure: float = None):

        # initilize
        self.logger = logging.getLogger("nanoCapture{}".format(camera_num))

        # populate desired settings from configuration file or function call
        self.camera_num = camera_num
        if exposure is not None: self._exposure   = exposure
        else:                    self._exposure   = configs['exposure']
        if res is not None:      self._camera_res = res
        else:                    self._camera_res = configs['camera_res']
        self._capture_width                       = self._camera_res[0] 
        self._capture_height                      = self._camera_res[1]
        self._display_res                         = configs['output_res']
        self._display_width                       = self._display_res[0]
        self._display_height                      = self._display_res[1]
        self._framerate                           = configs['fps']
        self._flip_method                         = configs['flip']

        # Threading Locks
        self.capture_lock    = Lock()
        self.frame_lock      = Lock()
        self.stopped         = True

        # open up the camera
        self._open_capture()

        # Init Frame and Thread
        self.frame     = None
        self.new_frame = False
        self.stopped   = False
        self.measured_fps = 0.0

        Thread.__init__(self)


    def _open_capture(self):
        """Open up the camera so we can begin capturing frames"""
        print(gstreamer_pipeline(
            capture_width  = self._capture_width,
            capture_height = self._capture_height,
            display_width  = self._display_width,
            display_height = self._display_height,
            framerate      = self._framerate,
            exposure_time  = self._exposure,
            flip_method    = self._flip_method)
            )

        self.capture = cv2.VideoCapture(gstreamer_pipeline(
            capture_width  = self._capture_width,
            capture_height = self._capture_height,
            display_width  = self._display_width,
            display_height = self._display_height,
            framerate      = self._framerate,
            exposure_time  = self._exposure,
            flip_method    = self._flip_method),
            cv2.CAP_GSTREAMER)

        self.capture_open = self.capture.isOpened()

        if not self.capture_open:
            # Appply Settings to camera
            self.logger.log(logging.CRITICAL, "Status:Failed to open camera!")

    #
    # Thread routines #################################################
    # Start Stop and Update Thread

    def stop(self): 
        """stop the thread"""
        self.stopped = True

    def start(self):
        """ set the thread start conditions """
        self.stopped = False
        T = Thread(target=self.update, args=())
        T.daemon = True # run in background
        T.start()

    # After Stating of the Thread, this runs continously
    def update(self):
        """ run the thread """
        last_fps_time = time.time()
        last_exposure_time = last_fps_time
        num_frames = 0
        while not self.stopped:
            current_time = time.time()
            # FPS calculation
            if (current_time - last_fps_time) >= 5.0: # update frame rate every 5 secs
                self.measured_fps = num_frames/5.0
                self.logger.log(logging.DEBUG, "Status:FPS:{}".format(self.measured_fps))
                num_frames = 0
                last_fps_time = current_time
            with self.capture_lock:
                _, img = self.capture.read()
            # set the frame var to the img we just captured
            self.frame = img
            # tell any threads waiting for a new frame that we have one
            num_frames += 1

            if self.stopped:
                self.capture.release()

    #
    # Frame routines ##################################################
    # Each camera stores frames locally

    @property
    def frame(self):
        """ returns most recent frame """
        self._new_frame = False
        return self._frame

    @frame.setter
    def frame(self, img):
        """ set new frame content """
        with self.frame_lock:
            self._frame = img
            self._new_frame = True

    @property
    def new_frame(self):
        """ check if new frame available """
        out = self._new_frame
        return out

    @new_frame.setter
    def new_frame(self, val):
        """ override wether new frame is available """
        self._new_frame = val

    #
    # Camera Routines ################################################
    #

    # OpenCV interface
    # Works for Sony IX219
    #cap.get(cv2.CAP_PROP_BRIGHTNESS)
    #cap.get(cv2.CAP_PROP_CONTRAST)
    #cap.get(cv2.CAP_PROP_SATURATION)
    #cap.get(cv2.CAP_PROP_HUE)
    #cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    #cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    #cap.get(cv2.CAP_PROP_FPS)

    #V4L2 interface
    #Works for Sonty IX219
    #v4l2-ctl --set-ctrl exposure= 13..683709
    #v4l2-ctl --set-ctrl gain= 16..170
    #v4l2-ctl --set-ctrl frame_rate= 2000000..120000000
    #v4l2-ctl --set-ctrl low_latency_mode=True
    #v4l2-ctl --set-ctrl bypass_mode=Ture
    #os.system("v4l2-ctl -c exposure_absolute={} -d {}".format(val,self.camera_num))

    # Read properties

    @property
    def exposure(self):              
        if self.capture_open:
            return self.capture._exposure                       
        else: return float("NaN")

    # Write

    @exposure.setter
    def exposure(self, val):
        if val is None:
            return
        val = int(val)
        self._exposure = val
        if self.capture_open:
            with self.capture_lock:
                os.system("v4l2-ctl -c exposure_absolute={} -d {}".format(val, self.camera_num))
            self.logger.log(logging.DEBUG, "Status:Exposure:{}".format(self._exposure))
        else:
            self.logger.log(logging.CRITICAL, "Status:Failed to set exposure to{}!".format(val))

###############################################################################
# Testing
###############################################################################

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    print("Starting Capture")
    camera = nanoCapture()
    camera.start()

    print("Getting Frames")
    window_handle = cv2.namedWindow("Nano CSI Camera", cv2.WINDOW_AUTOSIZE)
    while cv2.getWindowProperty("Nano CSI Camera", 0) >= 0:
        if camera.new_frame:
            cv2.imshow('Nano CSI Camera', camera.frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.stop()
    cv2.destroyAllWindows()
