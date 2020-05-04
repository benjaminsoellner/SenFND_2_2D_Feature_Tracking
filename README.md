# Sensor Fusion Engineer Project 2 - 2D Feature Tracking

## Benjamin Söllner

This project is forked from the [Udacity Sensor Fusion Nanodegree](https://www.udacity.com/course/sensor-fusion-engineer-nanodegree--nd313) online class content and subsequently completed to meet the courses project submission standards. The remaining section of this `README` includes the Reflection which has to be completed as part of this project and details about the general course content and how to build this project. The source code in this repo also contains the lesson quizzes in the separate ``src/quizzes`` folder. Go to [udacity/SFND_2D_Feature_Tracking](https://github.com/udacity/SFND_2D_Feature_Tracking) if you want to retrieve the original (unfinished) repo. Don't you cheat by copying my repo in order to use it as your Nanodegree submission! :-o

## Reflection

This section answers how this Udacity project submission fulfils the project [rubric](https://review.udacity.com/#!/rubrics/2549/view).

### MP.1 Data Buffer Optimization

A ring buffer is implemented by processing the images in a ``std::deque`` which is kept of maximum size ``2`` by using ``push_back`` in addition to ``pop_front``.

### MP.2 Keypoint Detection

``HARRIS`` and ``SHITOMASI`` are implemented by own methods ``detKeypointsHarris`` using ``cv::xfeatures2d::HarrisLaplaceFeatureDetector`` and
``detKeypointsShiTomasi`` using ``cv::goodFeaturesToTrack`` respectively. ``FAST``, ``BRISK``, ``ORB``, ``AKAZE``, and ``SIFT`` are implemented in 
``detKeypointsModern``. All detectors are looped over in one main loop that executes all Detector / Descriptor combinations.

### MP.3 Keypoint Removal

All keypoints outside the rect ``cv::Rect vehicleRect(535, 180, 180, 150)`` are discarded by checking found keypoints against this rect with 
``vehicleRect.contains(keypoint.pt)``.

### MP.4 Keypoint Descriptors

``BRIEF``, ``ORB``, ``FREAK``, ``AKAZE`` and ``SIFT`` are implemented in ``descKeypoints``. All detectors are looped over in one main loop that executes all 
Detector / Descriptor combinations.

### MP.5 Descriptor Matching

Descriptor matching is set to BF-matching by default. FLANN-matching is implemented in ``matchDescriptors`` an can be activated by setting ``matcherType`` to
``"MAT_FLANN"``.

### MP.6 Descriptor Distance Ratio

k-Nearest-Neighbor matching is implemented in ``matchDescriptors`` and set by default (``selectorType = "SEL_KNN"``). 

### MP.7 Performance Evaluation 1

Distribution / neighborhood size for all detectors:

**Shi-Tomasi** is very fast and has normal number of Keypoints:
```
[MP.7] SHITOMASI detection with n=1370 keypoints in 26.6425 ms
[MP.7] SHITOMASI detection with n=1301 keypoints in 18.2123 ms
[MP.7] SHITOMASI detection with n=1361 keypoints in 18.786 ms
[MP.7] SHITOMASI detection with n=1358 keypoints in 19.2025 ms
[MP.7] SHITOMASI detection with n=1333 keypoints in 18.3264 ms
[MP.7] SHITOMASI detection with n=1284 keypoints in 18.5922 ms
[MP.7] SHITOMASI detection with n=1322 keypoints in 18.4468 ms
[MP.7] SHITOMASI detection with n=1366 keypoints in 18.301 ms
[MP.7] SHITOMASI detection with n=1389 keypoints in 18.8334 ms
[MP.7] SHITOMASI detection with n=1339 keypoints in 18.608 ms
```

**Harris** takes long hand has small number of keypoints:
```
[MP.7] HARRIS detection with n= 683 keypoints in 318.747 ms
[MP.7] HARRIS detection with n= 677 keypoints in 178.026 ms
[MP.7] HARRIS detection with n= 698 keypoints in 233.515 ms
[MP.7] HARRIS detection with n= 641 keypoints in 196.036 ms
[MP.7] HARRIS detection with n= 665 keypoints in 112.791 ms
[MP.7] HARRIS detection with n= 641 keypoints in 147.554 ms
[MP.7] HARRIS detection with n= 634 keypoints in 125.437 ms
[MP.7] HARRIS detection with n= 670 keypoints in 131.448 ms
[MP.7] HARRIS detection with n= 672 keypoints in 119.778 ms
[MP.7] HARRIS detection with n= 654 keypoints in 131.338 ms
```

**FAST** is very fast and has normal number of keypoints:
```
[MP.7] FAST detection with n= 1824 keypoints in 1.0826 ms
[MP.7] FAST detection with n= 1832 keypoints in 0.9738 ms
[MP.7] FAST detection with n= 1810 keypoints in 1.0378 ms
[MP.7] FAST detection with n= 1817 keypoints in 1.1291 ms
[MP.7] FAST detection with n= 1793 keypoints in 0.9185 ms
[MP.7] FAST detection with n= 1796 keypoints in 0.95 ms
[MP.7] FAST detection with n= 1788 keypoints in 0.9306 ms
[MP.7] FAST detection with n= 1695 keypoints in 0.924 ms
[MP.7] FAST detection with n= 1749 keypoints in 0.9414 ms
[MP.7] FAST detection with n= 1770 keypoints in 0.9485 ms
```

**BRISK** is quite fast and has high number of keypoints:
```
[MP.7] BRISK detection with n= 2757 keypoints in 34.1809 ms
[MP.7] BRISK detection with n= 2777 keypoints in 37.8902 ms
[MP.7] BRISK detection with n= 2741 keypoints in 31.1003 ms
[MP.7] BRISK detection with n= 2735 keypoints in 29.7528 ms
[MP.7] BRISK detection with n= 2757 keypoints in 31.5168 ms
[MP.7] BRISK detection with n= 2695 keypoints in 30.5546 ms
[MP.7] BRISK detection with n= 2715 keypoints in 49.9263 ms
[MP.7] BRISK detection with n= 2628 keypoints in 31.3857 ms
[MP.7] BRISK detection with n= 2639 keypoints in 31.5352 ms
[MP.7] BRISK detection with n= 2672 keypoints in 29.9453 ms
```

**ORB** is very fast - number of keypoints is limited by program code:
```
[MP.7] ORB detection with n= 500 keypoints in 18.6313 ms
[MP.7] ORB detection with n= 500 keypoints in 10.2931 ms
[MP.7] ORB detection with n= 500 keypoints in 7.2637 ms
[MP.7] ORB detection with n= 500 keypoints in 7.414 ms
[MP.7] ORB detection with n= 500 keypoints in 6.3345 ms
[MP.7] ORB detection with n= 500 keypoints in 6.2275 ms
[MP.7] ORB detection with n= 500 keypoints in 7.1853 ms
[MP.7] ORB detection with n= 500 keypoints in 7.3766 ms
[MP.7] ORB detection with n= 500 keypoints in 6.5874 ms
[MP.7] ORB detection with n= 500 keypoints in 5.9397 ms
```

**AKAZE** is fast with normal number of keypoints:
```
[MP.7] AKAZE detection with n= 1351 keypoints in 53.0769 ms
[MP.7] AKAZE detection with n= 1327 keypoints in 44.6989 ms
[MP.7] AKAZE detection with n= 1311 keypoints in 45.0812 ms
[MP.7] AKAZE detection with n= 1351 keypoints in 48.9078 ms
[MP.7] AKAZE detection with n= 1360 keypoints in 55.8153 ms
[MP.7] AKAZE detection with n= 1347 keypoints in 70.2033 ms
[MP.7] AKAZE detection with n= 1363 keypoints in 47.6782 ms
[MP.7] AKAZE detection with n= 1331 keypoints in 44.9465 ms
[MP.7] AKAZE detection with n= 1358 keypoints in 39.4706 ms
[MP.7] AKAZE detection with n= 1331 keypoints in 54.5994 ms
```

**SIFT** is fast with normal number of keypoints:
```
[MP.7] SIFT detection with n= 1438 keypoints in 76.7562 ms
[MP.7] SIFT detection with n= 1371 keypoints in 82.5069 ms
[MP.7] SIFT detection with n= 1380 keypoints in 98.3211 ms
[MP.7] SIFT detection with n= 1335 keypoints in 72.9206 ms
[MP.7] SIFT detection with n= 1305 keypoints in 82.1444 ms
[MP.7] SIFT detection with n= 1369 keypoints in 82.9079 ms
[MP.7] SIFT detection with n= 1396 keypoints in 76.5143 ms
[MP.7] SIFT detection with n= 1382 keypoints in 87.5701 ms
[MP.7] SIFT detection with n= 1463 keypoints in 70.1024 ms
[MP.7] SIFT detection with n= 1422 keypoints in 74.4489 ms
```

### MP.8 / MP.9 Performance Evaluation 2 & 3

Here is the total number of matches (kNN-filtered & distance ratio tested, sum of all 9 image pairs) as well as the 
average run time per image pair for each algorithm, in ascending order by average runtime:

```
[MP.8/.9] (FAST, BRIEF) with total 1099 matches in average 3.05738 ms
[MP.8/.9] (FAST, ORB) with total 1081 matches in average 7.39321 ms
[MP.8/.9] (ORB, BRIEF) with total 545 matches in average 7.49819 ms
[MP.8/.9] (ORB, ORB) with total 761 matches in average 21.3168 ms
[MP.8/.9] (SHITOMASI, BRIEF) with total 944 matches in average 33.5233 ms
[MP.8/.9] (SHITOMASI, ORB) with total 907 matches in average 36.3086 ms
[MP.8/.9] (FAST, FREAK) with total 881 matches in average 38.1176 ms
[MP.8/.9] (ORB, FREAK) with total 421 matches in average 45.361 ms
[MP.8/.9] (SHITOMASI, FREAK) with total 766 matches in average 45.7368 ms
[MP.8/.9] (AKAZE, BRIEF) with total 1266 matches in average 48.2262 ms
[MP.8/.9] (AKAZE, ORB) with total 1186 matches in average 72.1576 ms
[MP.8/.9] (AKAZE, FREAK) with total 1188 matches in average 83.0418 ms
[MP.8/.9] (SIFT, BRIEF) with total 702 matches in average 96.1076 ms
[MP.8/.9] (AKAZE, AKAZE) with total 1259 matches in average 115.601 ms
[MP.8/.9] (HARRIS, BRIEF) with total 514 matches in average 146.538 ms
[MP.8/.9] (HARRIS, FREAK) with total 398 matches in average 165.024 ms
[MP.8/.9] (HARRIS, ORB) with total 512 matches in average 169.828 ms
[MP.8/.9] (SIFT, FREAK) with total 596 matches in average 178.495 ms
[MP.8/.9] (FAST, BRISK) with total 899 matches in average 239.518 ms
[MP.8/.9] (ORB, BRISK) with total 751 matches in average 255.748 ms
[MP.8/.9] (AKAZE, BRISK) with total 1215 matches in average 289.999 ms
[MP.8/.9] (SHITOMASI, BRISK) with total 767 matches in average 302.948 ms
[MP.8/.9] (BRISK, FREAK) with total 1526 matches in average 307.337 ms
[MP.8/.9] (BRISK, BRIEF) with total 1704 matches in average 314.065 ms
[MP.8/.9] (BRISK, ORB) with total 1510 matches in average 331.681 ms
[MP.8/.9] (SIFT, BRISK) with total 592 matches in average 339.366 ms
[MP.8/.9] (HARRIS, BRISK) with total 418 matches in average 477.621 ms
[MP.8/.9] (BRISK, BRISK) with total 1570 matches in average 519.785 ms
```

The following (Detector, Descriptor) implementations are most efficient w.r.t. runtime (in descending order):

* (FAST, BRIEF)
* (FAST, ORB)
* (ORB, BRIEF)

## Cource Content

<img src="images/keypoints.png" width="820" height="248" />

The idea of the camera course is to build a collision detection system - that's the overall goal for the Final Project. As a preparation for this, you will now build the feature tracking part and test various detector / descriptor combinations to see which ones perform best. This mid-term project consists of four parts:

* First, you will focus on loading images, setting up data structures and putting everything into a ring buffer to optimize memory load. 
* Then, you will integrate several keypoint detectors such as HARRIS, FAST, BRISK and SIFT and compare them with regard to number of keypoints and speed. 
* In the next part, you will then focus on descriptor extraction and matching using brute force and also the FLANN approach we discussed in the previous lesson. 
* In the last part, once the code framework is complete, you will test the various algorithms in different combinations and compare them with regard to some performance measures. 

See the classroom instruction and code comments for more details on each of these parts. Once you are finished with this project, the keypoint matching part will be set up and you can proceed to the next lesson, where the focus is on integrating Lidar points and on object detection using deep-learning. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./2D_feature_tracking`.