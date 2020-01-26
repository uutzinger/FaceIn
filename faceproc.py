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
import platform

# Open Face Detection and Image Processing
import face_recognition
import dlib
import cv2


class faceProc(Thread):
    """
    This thread continually processes frames
    """

    # Initialize the Process Thread
    def __init__(self, down_sampling: float=0.25):
        # initialize 
        self.logger     = logging.getLogger("faceProc")


        if dlib.DLIB_USE_CUDA:
            self.model = "cnn"
        else:
            self.model = "hog"

        # Threading Locks, Events
        self.stopped         = True

        # Init Frame and Thread
        self.frame_lock = Lock()
        self.frame     = None
        self.new_frame = False
        self.face_locations = []
        self.new_face_locations = False
        self.faces_present = False
        self.face_encodings = []
        self.stopped   = False
        self.measured_fps = 0.0
        self.down_sampling = down_sampling

        Thread.__init__(self)

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
                self.logger.log(logging.DEBUG, "Status:FacePS:{}".format(self.measured_fps))
                num_frames = 0
                last_fps_time = current_time

            # deal with new frame
            if self.new_frame:
                # Resize frame of video for faster face recognition processing
                small_frame = cv2.resize(self.frame, (0, 0), fx=self.down_sampling, fy=self.down_sampling) # <1ms
                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)            
                 # rgb_small_frame = small_frame[:, :, ::-1] # 0.02ms
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                self.face_locations = face_recognition.face_locations(rgb_small_frame, model=self.model) # 50-120ms 
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self._face_locations) # 500ms
                num_frames += 1
            
            time.sleep(0.005)

    # Service Routines
    ###################################################################

    # Frace locations
    #################

    @property
    def face_locations(self):
        """ returns most recent faces """
        self._new_face_locations = False
        return self._face_locations

    @face_locations.setter
    def face_locations(self, face_locations):
        """ set new faces """
        if not face_locations == []:
            self._face_locations = face_locations
            self._new_face_locations = True
            self._faces_present = True
        else:
            self._faces_present = False

    @property
    def new_face_locations(self):
        """ check if new faces available """
        out = self._new_face_locations
        return out

    @new_face_locations.setter
    def new_face_locations(self,val):
        """ override wether new faces are available """
        self._new_face_locations = val

    @property
    def faces_present(self):
        """ check if new faces available """
        out = self._faces_present
        return out

    @faces_present.setter
    def faces_present(self,val):
        """ override wether faces are in frame """
        self._faces_present = val

    # Face Encodings
    ################

    @property
    def face_encodings(self):
        """ returns most recent face encodigs """
        return self._face_encodings

    @face_encodings.setter
    def face_encodings(self, face_encodings):
        """ set new face encodings """
        self._face_encodings = face_encodings

    # Frame routines
    ################

    @property
    def frame(self):
        """ returns most recent frame """
        self._new_frame = False
        return self._frame

    @frame.setter
    def frame(self, img):
        """ set new frame content """
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

