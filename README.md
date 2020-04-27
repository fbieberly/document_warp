# Document warp
A script for performing perspective transforms on images of documents to give a head-on view of them.

## Description

This script is designed to take an image of a document and perform a perspective transform to produce an output image as the document would appear from a head-on perspective.  

The script differs slightly from the standard [perspective transform tutorials] in that it can warp shapes that are non-convex contours with more than 4 points (for example a document that has a sharp fold in it). This only works if a contour is detected around the document.  

[perspective transform tutorials]: https://docs.opencv.org/3.4/da/d6e/tutorial_py_geometric_transformations.html

## Example picture

Here is the input image:  
![input.jpg](https://github.com/fbieberly/document_warp/wiki/input.jpg?raw=true "input.jpg")

And here is the output image:  
![output.jpg](https://github.com/fbieberly/document_warp/wiki/output.jpg?raw=true "output.jpg")



## Dependencies

Should work with either Python 2.X or 3.X

Requires that you have the following packages installed:  
[OpenCV] for the image processing.  
[NumPy] and [SciPy] for the math and data types.  

pip install opencv-python numpy scipy

[OpenCV]: https://opencv.org/
[NumPy]: https://www.numpy.org/
[SciPy]: https://www.scipy.org/

## Getting started

1. Launch the script
    1. If you use the command line you can provide two arguments (input_file and output_file)
    1. ex. python document_warp.py input_file.jpg
    1. ex. python document_warp.py input_file.jpg output_file.jpg
1. Use the mouse to left-click on 4 points that are close to the corners of the document in your image.
    1. It's best if you click on points that are outside the corners of the document
    1. Red lines will be drawn, it's OK if they cross over parts of the document.
    1. Right-click to clear all the points you've selected.
1. Hit the 'a' key to try to perform automatic selection of a bounding contour.
    1. The automatic selection does two things: gets 4 corner points that are on the bounding contour (for a regular perspective transform) and finds a contour around the document (for a grid perspective transform).
    1. Automatic selection will fail if the script cannot detect a contour around the document. High contrast images of the document against a background will help.
1. Hit 'w' for a 'normal' perspective transform. This works well on documents that are smooth and flat. It will work regardless of whether automatic selection was able to find corners or a contour around the document.
1. Hit 'g' for a 'grid' perspective transform.
    1. This only works if automatic selection was able to find a contour. 
    1. In the script you can change the number of points in the grid (more points takes noticibly longer).
    1. The script doesn't try to adjust color/darkness so the output image will probably still look slightly bent because of the lighting of the original image.
1. Hit 's' to save the output image (defaults to 'warp_output.jpg')
    1. Only works while the output image is visible.

