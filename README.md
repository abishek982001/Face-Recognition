# Face-Recognition
This python program can reconize your face!
* This program just needs a single photo of yours.
* Add your photo in the `Images directory` or you can use `any other floders` in any location in your PC.
* Run the `main.py` file.
* Once you run `main.py`, you will be asked to enter the `path`.
* You should enter the correct path or else the program terminates. 
* The image you use should have proper names eg:`yourname.jpg/png` because the filename is assumed as the `person's name` by this program.
* If everything is right, this opens your `webcam`.
* Once your face is detected, a green color bounding box will appear along with the name which you used in the photo. 

# Requirements
Libraries used:
* [opencv](https://github.com/opencv/opencv) 
* [dlib](https://pypi.org/project/dlib/)
* [face_recognition](https://pypi.org/project/face-recognition/) library is a part of the [dlib](https://pypi.org/project/dlib/) library. But         some `wheel version` requires separate installation.
* [numpy](https://pypi.org/project/numpy/)