# Indian_Coin_Detection_&_Recognition

This projects focuses on detection and recognition of indian rupee coins in different and challenging environment conditions.
Firstly a image dataset of coins of denomination 1,2,5,10 are collected with different environment conditions and also the standard
images from google images. Next SIFT features of collected images are stored as knowledge base. Calculating the similarity score 
using FLANN matcher from OpenCV to match the SIFT descriptors of test and images with available ground truth.  

## Getting Started

To test with new images provide the file path as argument to execution command of respective scripts(Recognize and Detection). 

### Prerequisites

OpenCV,scipy and skimage.

