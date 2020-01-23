# Face In
Sign in system with face recognition

This program is based on doorcam.py from Adam Geitgey

htps://medium.com/@ageitgey/build-a-hardware-based-face-recognition-system-for-150-with-the-nvidia-jetson-nano-and-python-a25cb8c891fd

https://github.com/ageitgey/face_recognition#python-code-examples

Multithreaded camera drivers are provided for
- USB or builtin cameras using cv2 interface
- Raspberry Pi CSI camera using PiCamera
- Jetson Nano CSI camera using gstreamer
Camera settings are stored in configs.py

 
# To run this program you will need to following items insalled
  
pip3 install dlib
pip3 install numpy
pip3 install face_recognition
pip3 install screeninfo
On windows dlib needs a c compiler and cmake

Urs Utiznger
2019, 2020


This code used PyVmMonitor http://pyvmmonitor.com for profiling.
