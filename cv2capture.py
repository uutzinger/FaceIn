###############################################################################
# OpenCV video capture
# Uses opencv video capture to capture system's camera
# Adapts to operating system and allows configuation of codec
# BitBuckets FRC 4183
# 2019
###############################################################################

###############################################################################
# Imports
###############################################################################

# Multi Threading
from threading import Thread
from threading import Lock
from threading import Event

#
import logging
import time
import sys

# Open Computer Vision
import cv2

# Camera configuration file
from configs   import configs

###############################################################################
# Video Capture
###############################################################################

class cv2Capture(Thread):
    """
    This thread continually captures frames from a USB camera
    """

    # Initialize the Camera Thread
    # Opens Capture Device and Sets Capture Properties
    def __init__(self, camera_num: int = 0, res: (int, int) = None, 
                 exposure: float = None):
        # initialize 
        self.logger     = logging.getLogger("cv2Capture{}".format(camera_num))

        # populate desired settings from configuration file or function call
        self.camera_num = camera_num
        if exposure is not None: self._exposure   = exposure
        else:                    self._exposure   = configs['exposure']
        if res is not None:      self._camera_res = res
        else:                    self._camera_res = (configs['camera_res'])
        self._display_res                         = configs['output_res']
        self._display_width                       = self._display_res[0]
        self._display_height                      = self._display_res[1]
        self._framerate                           = configs['fps']
        self._flip_method                         = configs['flip']
        self._buffersize                          = configs['buffersize']         # camera drive buffer size
        self._fourcc                              = configs['fourcc']             # camera sensor encoding format

        if self._exposure <= 0:
            self._autoexposure = True
        else:
            self._autoexposure = False

        # Threading Locks, Events
        self.capture_lock    = Lock() # before changing capture settings lock them
        self.frame_lock      = Lock() # When copying frame, lock it
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
        # Open the camera with platform optimal settings
        if sys.platform.startswith('win'):
            self.capture = cv2.VideoCapture(self.camera_num, apiPreference=cv2.CAP_MSMF)
        elif sys.platform.startswith('darwin'):
            self.capture = cv2.VideoCapture(self.camera_num, apiPreference=cv2.CAP_AVFOUNDATION)
        elif sys.platform.startswith('linux'):
            self.capture = cv2.VideoCapture(self.camera_num, apiPreference=cv2.CAP_V4L2)
        else:
            self.capture = cv2.VideoCapture(self.camera_num, apiPreference=cv2.CAP_ANY)

        self.capture_open = self.capture.isOpened()

        # self.cv2SettingsDebug() # check camera properties

        if self.capture_open:
            # Apply settings to camera
            self.height       = self._camera_res[1]           # image resolution
            self.width        = self._camera_res[0]           # image resolution
            self.autoexposure = self._autoexposure            # autoexposure
            self.exposure     = self._exposure                # camera exposure
            self.buffersize   = self._buffersize               # camera drive buffer size
            self.fourcc       = self._fourcc                  # camera sensor encoding format
            self.fps          = self._framerate               # desired fps
        else:
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
            if img is not None:
                if self._display_height > 0:
                    # tmp = cv2.resize(img, self._display_res, interpolation = cv2.INTER_NEAREST)
                    tmp = cv2.resize(img, self._display_res)
                else:
                    tmp = img

                if   self._flip_method == 1: # ccw 90
                    self.frame = cv2.roate(tmp, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif self._flip_method == 2: # rot 180, same as flip lr & up
                    self.frame = cv2.roate(tmp, cv2.ROTATE_180)
                elif self._flip_method == 3: # cw 90
                    self.frame = cv2.roate(tmp, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif self._flip_method == 4: # horizontal
                    self.frame = cv2.flip(tmp, 0)
                elif self._flip_method == 5: # upright diagonal. ccw & lr
                    tmp = cv2.roate(tmp, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    self.frame = cv2.flip(tmp, 1)
                elif self._flip_method == 6: # vertical
                    self.frame = cv2.flip(tmp, 1)
                elif self._flip_method == 7: # upperleft diagonal
                    self.frame = cv2.transpose(tmp)
                else:
                    self.frame = tmp

                num_frames += 1

            if self.stopped:
                self.capture.release()        

    #
    # Frame routines ##################################################
    # Each camera stores frames locally
    ###################################################################

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
    # Camera routines ##################################################
    # Rading and Setting Camera Options
    #

    @property
    def width(self):
        """ returns video capture width """
        if self.capture_open:
            return self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        else: return float("NaN")

    @width.setter
    def width(self, val):
        """ sets video capture width """
        if (val is None) or (val == -1):
            self.logger.log(logging.DEBUG, "Status:Width not changed:{}".format(val))
            return
        if self.capture_open:
            with self.capture_lock: 
                isok = self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, val)
            if isok:
                self.logger.log(logging.DEBUG, "Status:Width:{}".format(val))
            else:
                self.logger.log(logging.CRITICAL, "Status:Failed to set width to {}!".format(val))

    @property
    def height(self):
        """ returns videocapture height """
        if self.capture_open:
            return self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        else: return float("NaN")

    @height.setter
    def height(self, val):
        """ sets video capture height """
        if (val is None) or (val == -1):
            self.logger.log(logging.DEBUG, "Status:Height not changed:{}".format(val))
            return
        if self.capture_open:
            with self.capture_lock: 
                isok = self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(val))
            if isok:
                self.logger.log(logging.DEBUG, "Status:Height:{}".format(val))
            else:
                self.logger.log(logging.CRITICAL, "Status:Failed to set height to {}!".format(val))

    @property
    def resolution(self):
        """ returns current resolution width x height """
        if self.capture_open:
            return [self.capture.get(cv2.CAP_PROP_FRAME_WIDTH), 
                    self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)]
        else: return [float("NaN"), float("NaN")] 

    @resolution.setter
    def resolution(self, val):
        if val is None: return
        if self.capture_open:
            if len(val) > 1: # have width x height
                with self.capture_lock: 
                    isok0 = self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(val[0]))
                    isok1 = self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(val[1]))
                if isok0 and isok1:
                    self.logger.log(logging.DEBUG, "Status:Width:{}".format(val[0]))
                    self.logger.log(logging.DEBUG, "Status:Height:{}".format(val[1]))
                else:
                    self.logger.log(logging.CRITICAL, "Status:Failed to set resolution to {},{}!".format(val[0],val[1]))
            else: # given only one value for resolution
                with self.capture_lock: 
                    isok0 = self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(val))
                    isok1 = self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(val))
                if isok0 and isok1:
                    self.logger.log(logging.DEBUG, "Status:Width:{}".format(val))
                    self.logger.log(logging.DEBUG, "Status:Height:{}".format(val))
                else:
                    self.logger.log(logging.CRITICAL, "Status:Failed to set resolution to {},{}!".format(val,val))
        else: # camera not open
            self.logger.log(logging.CRITICAL, "Status:Failed to set resolution, camera not open!")

    @property
    def exposure(self):
        """ returns curent exposure """
        if self.capture_open:
            return self.capture.get(cv2.CAP_PROP_EXPOSURE)
        else: return float("NaN")

    @exposure.setter
    def exposure(self, val):
        """ # sets current exposure """
        self._exposure = val
        if (val is None) or (val == -1):
            self.logger.log(logging.CRITICAL, "Status:Can not set exposure to {}!".format(val))
            return
        if self.capture_open:
            with self.capture_lock:
                isok = self.capture.set(cv2.CAP_PROP_EXPOSURE, self._exposure)
            if isok:
                self.logger.log(logging.DEBUG, "Status:Exposure:{}".format(val))
            else:
                self.logger.log(logging.CRITICAL, "Status:Failed to set expsosure to:{}".format(val))

    @property
    def autoexposure(self):
        """ returns curent exposure """
        if self.capture_open:
            return self.capture.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        else: return float("NaN")

    @autoexposure.setter
    def autoexposure(self, val):
        """ sets autoexposure """
        if (val is None) or (val == -1):
            self.logger.log(logging.CRITICAL, "Status:Can not set Auto Exposure to:{}".format(val))
            return
        if self.capture_open:
            with self.capture_lock:
                isok = self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, val)
            if isok:
                self.logger.log(logging.DEBUG, "Status:Autoexpsure:{}".format(val))
            else:
                self.logger.log(logging.CRITICAL, "Status:Failed to set auto exposure to:{}".format(val))

    @property
    def fps(self):
        """ returns current frames per second setting """
        if self.capture_open:
            return self.capture.get(cv2.CAP_PROP_FPS)
        else: return float("NaN")

    @fps.setter
    def fps(self, val):
        """ set frames per second in camera """
        if (val is None) or (val == -1):
            self.logger.log(logging.CRITICAL, "Status:Can not set framerate to:{}".format(val))
            return
        if self.capture_open:
            with self.capture_lock:
                isok = self.capture.set(cv2.CAP_PROP_FPS, val)
            if isok:
                self.logger.log(logging.DEBUG, "Status:FPS:{}".format(val))
            else:
                self.logger.log(logging.CRITICAL, "Status:Failed to set FPS to:{}".format(val))

    @staticmethod
    def decode_fourcc(val):
        """ decode the fourcc integer to the chracter string """
        return "".join([chr((int(val) >> 8 * i) & 0xFF) for i in range(4)])

    @property
    def fourcc(self):
        """ return video encoding format """
        if self.capture_open:
            self._fourcc = self.capture.get(cv2.CAP_PROP_FOURCC)
            self._fourcc_str = self.decode_fourcc(self._fourcc)
            return self._fourcc_str
        else: return "None"

    @fourcc.setter
    def fourcc(self, val):
        """ set video encoding format in camera """
        if (val is None) or (val == -1):
            self.logger.log(logging.CRITICAL, "Status:Can not set FOURCC to:{}!".format(val))
            return
        if isinstance(val, str):  # we need to convert from FourCC to integer
            self._fourcc     = cv2.VideoWriter_fourcc(val[0],val[1],val[2],val[3])
            self._fourcc_str = val
        else: # fourcc is integer/long
            self._fourcc     = val
            self._fourcc_str = self.decode_fourcc(val)
        if self.capture_open:
            with self.capture_lock: 
                isok = self.capture.set(cv2.CAP_PROP_FOURCC, self._fourcc)
            if isok :
                self.logger.log(logging.DEBUG, "Status:FOURCC:{}".format(self._fourcc_str))
            else:
                self.logger.log(logging.CRITICAL, "Status:Failed to set FOURCC to:{}".format(self._fourcc_str))

    @property
    def buffersize(self):
        """ return opencv camera buffersize """
        if self.capture_open:
            return self.capture.get(cv2.CAP_PROP_BUFFERSIZE)
        else: return float("NaN")

    @buffersize.setter
    def buffersize(self, val):
        """ set opencv camera buffersize """
        if val is None or val == -1:
            self.logger.log(logging.CRITICAL, "Status:Can not set buffer size to:{}".format(val))
            return
        if self.capture_open:
            with self.capture_lock:
                isok = self.capture.set(cv2.CAP_PROP_BUFFERSIZE, val)
            if isok:
                self.logger.log(logging.DEBUG, "Status:Buffersize:{}".format(val))
            else:
                self.logger.log(logging.CRITICAL, "Status:Failed to set buffer size to:{}".format(val))

    def cv2SettingsDebug(self):
        """ return opencv camera properties """
        if self.capture_open:
            print(self.capture.get(cv2.CAP_PROP_POS_MSEC))             # 0          ,-1
            print(self.capture.get(cv2.CAP_PROP_POS_FRAMES))           # NA ELP     ,-1
            print(self.capture.get(cv2.CAP_PROP_POS_AVI_RATIO))        # NA ELP     ,-1
            print(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))          # 320        ,640
            print(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))         # 240        ,480
            print(self.capture.get(cv2.CAP_PROP_FPS))                  # 120        ,0
            print(self.capture.get(cv2.CAP_PROP_FOURCC))               # MJPG       ,844715353
            tmp = self.decode_fourcc(self.capture.get(cv2.CAP_PROP_FOURCC))         
            print(tmp)                                                 #            ,YUY2
            print(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))          # NA ELP     ,-1
            print(self.capture.get(cv2.CAP_PROP_FORMAT))               # 16         ,-1
            print(self.capture.get(cv2.CAP_PROP_MODE))                 # MJPG       ,-1
            print(self.capture.get(cv2.CAP_PROP_BRIGHTNESS))           # 0.5        ,0
            print(self.capture.get(cv2.CAP_PROP_CONTRAST))             # 0.5        ,0
            print(self.capture.get(cv2.CAP_PROP_SATURATION))           # 0.46875    ,64
            print(self.capture.get(cv2.CAP_PROP_HUE))                  # 0.5        ,0
            print(self.capture.get(cv2.CAP_PROP_GAIN))                 # 0.0        ,-1.0
            print(self.capture.get(cv2.CAP_PROP_EXPOSURE))             # 1.0        ,-4.0
            print(self.capture.get(cv2.CAP_PROP_CONVERT_RGB))          # 1.0        ,1.0
            print(self.capture.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U)) # NA ELP     ,4600.0
            print(self.capture.get(cv2.CAP_PROP_RECTIFICATION))        # NA ELP     ,-1.0
            print(self.capture.get(cv2.CAP_PROP_MONOCHROME))           # NA ELP     ,-1.0
            print(self.capture.get(cv2.CAP_PROP_SHARPNESS))            # 2.0        ,2.0
            print(self.capture.get(cv2.CAP_PROP_AUTO_EXPOSURE))        # 0.25 or 0.75(auto) ,-1.0
            print(self.capture.get(cv2.CAP_PROP_GAMMA))                # 100.0      ,100.0
            print(self.capture.get(cv2.CAP_PROP_TEMPERATURE))          # 6500       ,-1
            print(self.capture.get(cv2.CAP_PROP_TRIGGER))              # NA ELP     ,-1
            print(self.capture.get(cv2.CAP_PROP_TRIGGER_DELAY))        # NA ELP     ,-1
            print(self.capture.get(cv2.CAP_PROP_WHITE_BALANCE_RED_V))  # NA ELP     ,-1
            print(self.capture.get(cv2.CAP_PROP_ZOOM))                 # NA ELP     ,-1
            print(self.capture.get(cv2.CAP_PROP_FOCUS))                # NA ELP     ,-1
            print(self.capture.get(cv2.CAP_PROP_GUID))                 # NA ELP     ,-1
            print(self.capture.get(cv2.CAP_PROP_ISO_SPEED))            # NA ELP     ,-1
            print(self.capture.get(cv2.CAP_PROP_BACKLIGHT))            # 1.0        ,3.0
            print(self.capture.get(cv2.CAP_PROP_PAN))                  # NA ELP     ,-1
            print(self.capture.get(cv2.CAP_PROP_TILT))                 # NA ELP     ,-1
            print(self.capture.get(cv2.CAP_PROP_ROLL))                 # NA ELP     ,-1
            print(self.capture.get(cv2.CAP_PROP_IRIS))                 # NA ELP     ,-1
            print(self.capture.get(cv2.CAP_PROP_SETTINGS))             # NA ELP     ,0
            print(self.capture.get(cv2.CAP_PROP_BUFFERSIZE))           # 4.0        ,-1
            print(self.capture.get(cv2.CAP_PROP_AUTOFOCUS))            # NA ELP     ,-1
            print(self.capture.get(cv2.CAP_PROP_SAR_NUM))              # NA ELP     ,-1
            print(self.capture.get(cv2.CAP_PROP_SAR_DEN))              # NA ELP     ,-1
            print(self.capture.get(cv2.CAP_PROP_BACKEND))              # 200        ,700
            print(self.capture.get(cv2.CAP_PROP_CHANNEL))              # -1         ,0
            print(self.capture.get(cv2.CAP_PROP_AUTO_WB))              # 1.0        ,-1
            print(self.capture.get(cv2.CAP_PROP_WB_TEMPERATURE))       # 6500       ,-1
        else: 
            print("NaN")

###############################################################################
# Testing
###############################################################################

if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    print("Starting Capture")
    camera = cv2Capture(camera_num=0, res=(320,240))
    camera.start()

    print("Getting Frames")
    while True:
        if camera.new_frame:
            cv2.imshow('my webcam', camera.frame)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    camera.stop()
