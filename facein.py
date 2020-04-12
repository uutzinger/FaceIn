###############################################################################
# Face In 
# Recognizes visitors, records appearance time and display currently signed in
# visitors.
#
# This program is based on doorcam.py from Adam Geitgey
# https://medium.com/@ageitgey/build-a-hardware-based-face-recognition-system-for-150-with-the-nvidia-jetson-nano-and-python-a25cb8c891fd
# https://github.com/ageitgey/face_recognition#python-code-examples
#
# Multithreaded camera drivers are provided for
# - USB or builtin cameras using cv2 interface
# - Raspberry Pi CSI camera using PiCamera
# - Jetson Nano CSI camera using gstreamer
# Camera settings are stored in configs.py
# 
# To run this program you will need to following items insalled
#  
# pip3 install dlib
# pip3 install numpy
# pip3 install face_recognition
# pip3 install screeninfo
# On windows dlib needs a c compiler and cmake
#
# Urs Utiznger
# 2019, 2020
###############################################################################

###############################################################################
# Imports
###############################################################################
import numpy as np
import platform
import pickle
import logging
import time
import cv2
import face_recognition
import logging

from   datetime   import datetime, timedelta
from   screeninfo import get_monitors
from   faceproc   import faceProc

###############################################################################
# Intializing
###############################################################################

# Our list of known face encodings and a matching list of metadata about each face.
known_face_encodings = []
known_face_metadata = []

# Figure out ideal window size
monitor = get_monitors()
display_width   = int(monitor[0].width * 0.8)
display_height  = int(monitor[0].height * 0.8)
face_size       = 150
time_history    = 60 # minutes
max_display_interval = 0.030

datafile = "210_known_faces.dat"

###############################################################################
# Functions
###############################################################################

def save_known_faces():
    """
    Stores the face meta data and face encodings on disk
    """
    with open(datafile, "wb") as face_data_file:
        face_data = [known_face_encodings, known_face_metadata]
        pickle.dump(face_data, face_data_file)
        print("Known faces backed up to disk.")

def load_known_faces():
    """ 
    Load the face meta data and encodings from disk
    """
    global known_face_encodings, known_face_metadata
    try:
        with open(datafile, "rb") as face_data_file:
            known_face_encodings, known_face_metadata = pickle.load(face_data_file)
            print("Known faces loaded from disk.")
    except FileNotFoundError as e:
        print("No previous face data found - starting with a blank known face list.")
        pass

def register_new_face(face_encoding, face_image):
    """
    Add a new person to our list of known faces
    """
    # Add the face encoding to the list of known faces
    known_face_encodings.append(face_encoding)
    # Add a matching dictionary entry to our metadata list.
    # We can use this to keep track of how many times a person has visited, when we last saw them, etc.
    known_face_metadata.append({
        "first_seen": datetime.now(),
        "first_seen_this_interaction": datetime.now(),
        "last_seen": datetime.now(),
        "registrations": [datetime.now()],
        "seen_count": 1,
        "seen_frames": 1,
        "face_image": face_image,
        "First": "NoName",
        "Last": "NoName",
    })

def lookup_known_face(face_encoding):
    """
    See if this is a face we already have in our face list
    """
    metadata = None

    # If our known face list is empty, just return nothing since we can't possibly have seen this face.
    if len(known_face_encodings) == 0:
        return metadata

    # Calculate the face distance between the unknown face and every face on in our known face list
    # This will return a floating point number between 0.0 and 1.0 for each known face. The smaller the number,
    # the more similar that face was to the unknown face.
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

    # Get the known face that had the lowest distance (i.e. most similar) from the unknown face.
    best_match_index = np.argmin(face_distances)

    # If the face with the lowest distance had a distance under 0.6, we consider it a face match.
    # 0.6 comes from how the face recognition model was trained. It was trained to make sure pictures
    # of the same person always were less than 0.6 away from each other.
    # Here, we are loosening the threshold a little bit to 0.65 because it is unlikely that two very similar
    # people will come up to the door at the same time.
    if face_distances[best_match_index] < 0.6:
        # If we have a match, look up the metadata we've saved for it (like the first time we saw it, etc)
        metadata = known_face_metadata[best_match_index]

        # Update the metadata for the face so we can keep track of how recently we have seen this face.
        metadata["last_seen"] = datetime.now()
        metadata["seen_frames"] += 1

        # We'll also keep a total "seen count" that tracks how many times this person has come to the door.
        # But we can say that if we have seen this person within the last 5 minutes, it is still the same
        # visit, not a new visit. But if they go away for awhile and come back, that is a new visit.
        if datetime.now() - metadata["first_seen_this_interaction"] > timedelta(minutes=5):
            metadata["first_seen_this_interaction"] = datetime.now()
            metadata["seen_count"] += 1
            metadata["registrations"].append([datetime.now()])
            known_face_metadata[best_match_index]["registrations"].append([datetime.now()])

    return metadata

def create_face_labels():
    face_locations = face.face_locations
    face_encodings = face.face_encodings
    face_labels = []
    # 0.2ms
    # Loop through each detected face and see if it is one we have seen before
    # If so, we'll give it a label that we'll draw on top of the video.
    for face_location, face_encoding in zip(face_locations, face_encodings):
        # See if this face is in our list of known faces
        ################################################
        metadata = lookup_known_face(face_encoding)
        # If we found the face, label the face with some useful information.
        if metadata is not None:
            time_at_door = datetime.now() - metadata['first_seen_this_interaction']
            face_label = f"{metadata['First']} {int(time_at_door.total_seconds())}s"
        else: # this is a brand new face, add it to our list of known faces
            face_label = "New visitor!"
            # Grab the image of the the face from the current frame of video
            top, right, bottom, left = face_location
            width = right - left
            height = bottom - top
            #  make sure image for new face is at least the required size
            if height >= face_size:
                # scale to desired height
                face_image = cv2.resize( face.frame[top:bottom, left:right], 
                                            (int(width*face_size/height), face_size) )
                # Add the new face to our known face data
                register_new_face(face_encoding, face_image)
        face_labels.append(face_label)
    return face_labels


def update_display_frame_with_recent_visitors(display_frame, time_history, logger):
    """
    Creates an updated display image with recent vistor face snap shots
    """

    # We run on 4k display: this would be 3840 x 2160
    face_columns = int(display_width  / (face_size + 30))
    face_rows    = int(display_height / (face_size + 30))

    display_frame.fill(0) # clear display frame
    # initialize
    y_position = 30
    x_position = 0

    for metadata in known_face_metadata:
        # If we have seen this person in the last hour, draw their image
        if datetime.now() - metadata["last_seen"] < timedelta(minutes=time_history) and metadata["seen_frames"] > 1:
            
            # is the face location overlappping with video
            if ( y_position+face_size > (display_height - camera.height) ) \
                and ( x_position+face_size > (display_width - camera.width) ):
                x_position  = 0
                y_position += (face_size + 30)

            # Draw the known face image
            face_image = metadata["face_image"]
            display_frame[y_position:y_position+face_image.shape[0], x_position:x_position+face_image.shape[1]] = face_image

            # Label the image with how many times they have visited
            visits = metadata['seen_count']
            first_name = metadata['First']
            visit_label = f"{first_name} {visits} visits"
            if first_name is None or first_name == [] or first_name == "NoName":
                visit_label = f"No Name {visits}"
            cv2.putText(display_frame, visit_label, (x_position + 10, y_position + face_size + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            # update position for next image
            x_position += (face_size + 30)
            if (x_position+face_size+30) >= display_width: 
                x_position  = 0
                y_position += (face_size + 30)                
            if (y_position+face_size+30) >= display_height: 
                logger.log(logging.CRITICAL, "Status:Failed to display, not enough space")

    cv2.putText(display_frame, "Face In: Urs Utzinger 2020", ( 10, display_height - 35), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

    return display_frame

if __name__ == "__main__":
    """
    Main program
    """

    logging.basicConfig(level=logging.DEBUG)
    logger  = logging.getLogger("FaceIn")

    # reate camera interface
    plat = platform.system()
    if plat == 'Windows': 
        from cv2capture import cv2Capture
        camera = cv2Capture()
    elif plat == 'Linux':
        if platform.machine() == "aarch64":
            from nanocapture import nanoCapture
            camera = nanoCapture()
        elif platform.machine() == "armv6l":
            from picapture import piCapture
            camera = piCapture()
    elif plat == 'MacOS':
        from cv2capture import cv2Capture
        camera = cv2Capture()
    else:
        from cv2Capture import cv2Capture
        camera = cv2Capture()

    frame_height = int(camera.height)
    frame_width  = int(camera.width)

    if frame_height <= 720:
        down_sampling = 0.5
        up_sampling = 2
    else:
        down_sampling = 0.25
        up_sampling = 4

    # Create Face Processing Thread
    face = faceProc(down_sampling = down_sampling)

    # Magic begins here
    logger.log(logging.DEBUG, "Starting Capture Thread")
    camera.start()    
    load_known_faces()
    logger.log(logging.DEBUG, "Starting Face Recgnition Thread")
    face.start()

    # Preallocate display image and create display window
    display_frame = np.zeros((display_height,display_width,3), np.uint8)
    window_handle = cv2.namedWindow("Registration", cv2.WINDOW_AUTOSIZE)
    # AUTOSIZE (does not seem to create approprite size and enlarges and crops frame)
    # NORMAL 

    # Initialize variable
    last_display_time   = time.time()
    last_face_save_time = time.time()
    face_labels         = []
    face_locations      = []
    face_encodings      = []
    
    update_display_frame = False            # Is there new camera frame
    update_display_frame_visitors = False   # Are there new faces and we need to update visitor list

    while cv2.getWindowProperty("Registration", 0) >= 0:
        # Main Program Loop: Execute until window closed or users persses key
        ##############################################################################

        current_time = time.time()

        # Do we have new frames from the camera?
        ########################################
        if camera.new_frame:
            frame = camera.frame
            face.frame = frame
            update_display_frame = True

        # Do we have new face locations?
        ################################
        if face.new_face_locations:
            face_labels = create_face_labels()
            update_display_frame_visitors = True

        # Do we have new frame to display
        #################################
        if update_display_frame:

            # Do we have faces in our frames?
            #################################
            if face.faces_present:
                face_locations = face.face_locations
                face_encodings = face.face_encodings
                if face_labels == []:
                    face_labels = create_face_labels()
                # Draw a box around each face and label each face 0.1ms
                #######################################################
                for (top, right, bottom, left), face_label in zip(face_locations, face_labels):
                    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                    top    *= up_sampling
                    right  *= up_sampling
                    bottom *= up_sampling
                    left   *= up_sampling
                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    cv2.putText(frame, face_label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                    # Draw a dot on faceencoding locations

            # Update recent visitor display if needed
            #########################################
            if update_display_frame_visitors or current_time - last_display_time > (time_history * 60.0 / 3.0):
                display_frame = update_display_frame_with_recent_visitors(display_frame, time_history, logger)
                update_display_frame_visitors = False
                last_display_time = current_time

            # Udpate the display
            ####################
            display_frame[-frame_height-1:-1,-frame_width-1:-1,:] = frame # 0.5ms
            cv2.imshow('Registration', display_frame) # 7ms
            update_display_frame = False # done

        # We need to save our known faces back to disk 
        # every so often in case something crashes (20mins)
        ################################################
        if current_time - last_face_save_time > (time_history * 60.0 / 3.0) :
            save_known_faces()
            last_face_save_time =  current_time

        # Hit 'q' on the keyboard to quit!
        ##################################
        if cv2.waitKey(1) & 0xFF == ord('q'):
            save_known_faces()
            break

        # Limit frame rate on display
        #############################
        time_remaining = max_display_interval - (time.time() - current_time) 
        if time_remaining > 0.001: time.sleep(time_remaining)

    # Clean up and Release handle to window
    #######################################
    camera.stop()
    face.stop()
    cv2.destroyAllWindows()
