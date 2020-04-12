###############################################################################
# Add Directory
# Update known_faces.dat with images and data from given folder
#
# This program is based on doorcam.py from Adam Geitgey
# https://github.com/ageitgey/face_recognition#python-code-examples
## 
# To run this program you will need to following items insalled
#  
# pip3 install dlib
# pip3 install numpy
# pip3 install face_recognition
# pip3 install screeninfo
# On windows dlib needs a c compiler and cmake
#
# Urs Utiznger
# 2020
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
import argparse
import os
import re
import dlib
from datetime import datetime

###############################################################################
# Intializing
###############################################################################

# Our list of known face encodings and a matching list of metadata about each face.
known_face_encodings = []
known_face_metadata = []

# Desired face size
face_size       = 150
datafile = "known_faces.dat"


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
    if face_distances[best_match_index] <= 0.6:
        # If we have a match, look up the metadata we've saved for it (like the first time we saw it, etc)
        metadata = known_face_metadata[best_match_index]

    return metadata

def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png|jfif|tif|tiff)', f, flags=re.I)]

if __name__ == "__main__":
    """
    Main program
    """
    ap = argparse.ArgumentParser(description='Process some image(s).')
    ap.add_argument('-f', '--folder', dest='folder', default="./images", help='folder to search for images')
    args = vars(ap.parse_args())

    if dlib.DLIB_USE_CUDA: fmodel = "cnn"
    else:                  fmodel = "hog"

    # fmodel = "hog"
    # load_known_faces()

    window_handle      = cv2.namedWindow("Registration", cv2.WINDOW_AUTOSIZE)
    window_handle_face = cv2.namedWindow("Face", cv2.WINDOW_AUTOSIZE)

    for file in image_files_in_folder(args["folder"]):

        frame = face_recognition.load_image_file(file)
        basename = os.path.splitext(os.path.basename(file))[0]
        Name= basename.split()
        First_Name = Name[0] 
        Last_Name = Name[1]
        face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=2, model=fmodel)
        face_encodings = face_recognition.face_encodings(frame, face_locations, num_jitters=2)
        face_landmarks = face_recognition.face_landmarks(frame, face_locations, model='large')

        (top, right, bottom, left) = face_locations[0]
        face_encoding              = face_encodings[0]
        
        # metadata = lookup_known_face(face_encoding)

        width = right - left
        height = bottom - top
        face_image = cv2.resize( frame[top:bottom, left:right], (int(width*face_size/height), face_size) )
        face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)

        known_face_encodings.append(face_encoding)
        known_face_metadata.append({
            "first_seen": datetime.now(),
            "first_seen_this_interaction": datetime.now(),
            "last_seen": datetime.now(),
            "registrations": [datetime.now()],
            "seen_count": 0,
            "seen_frames": 0,
            "face_image": face_image,
            "First": First_Name,
            "Last": Last_Name,
        })

        cv2.imshow('Face', face_image)

        for face_landmark in face_landmarks:
            for facial_feature in face_landmark.keys():
                pts = np.array([face_landmark[facial_feature]], np.int32) 
                pts = pts.reshape((-1,1,2))
                cv2.polylines(frame, [pts], False, (255,255,255))

        cv2.imshow('Registration', cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)) # 7ms
        cv2.waitKey(0)

    save_known_faces()
cv2.destroyAllWindows()
