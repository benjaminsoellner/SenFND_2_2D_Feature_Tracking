/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <deque>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "./"; // TODO: only done for debugging
    //string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";

    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    deque<DataFrame> dataBuffer;  // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results    
    bool bLimitKpts = false;      // optional : limit number of keypoints (helpful for debugging and learning)

    vector<string> detectorTypes = { "SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT" };
    vector<string> descriptorTypes = { "BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT" };

    for (string detectorType: detectorTypes)
    {
        for (string descriptorType: descriptorTypes)
        {
            double totalTime = 0.0f;
            int totalMatches = 0;

            /* MAIN LOOP OVER ALL IMAGES */
            for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
            {

                /* LOAD IMAGE INTO BUFFER */
                // assemble filenames for current index
                ostringstream imgNumber;
                imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
                string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;
                // load image from file and convert to grayscale
                cv::Mat img, imgGray;
                img = cv::imread(imgFullFilename);
                cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
                //// STUDENT ASSIGNMENT
                //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize
                // push image into data frame buffer
                DataFrame frame;
                frame.cameraImg = imgGray;
                while (dataBuffer.size() > dataBufferSize) {
                    dataBuffer.pop_front();
                }
                dataBuffer.push_back(frame);
                //// EOF STUDENT ASSIGNMENT
                cout << "#1 load image into buffer done" << endl;

                double t = (double)cv::getTickCount();
                /* DETECT IMAGE KEYPOINTS */
                // extract 2D keypoints from current image
                vector<cv::KeyPoint> keypoints; // create empty feature list for current image
                //// STUDENT ASSIGNMENT
                //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
                //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
                if (detectorType.compare("SHITOMASI") == 0)
                {
                    detKeypointsShiTomasi(keypoints, imgGray, false);
                }
                else if (detectorType.compare("HARRIS") == 0)
                {
                    detKeypointsHarris(keypoints, imgGray, false);
                }
                else 
                {
                    detKeypointsModern(keypoints, imgGray, detectorType, false);
                }
                //// EOF STUDENT ASSIGNMENT
                //// STUDENT ASSIGNMENT
                //// TASK MP.3 -> only keep keypoints on the preceding vehicle
                // only keep keypoints on the preceding vehicle
                bool bFocusOnVehicle = true;
                vector<cv::KeyPoint> filteredKeypoints;
                cv::Rect vehicleRect(535, 180, 180, 150);
                if (bFocusOnVehicle)
                {
                    for (cv::KeyPoint keypoint: keypoints)
                    {
                        if (vehicleRect.contains(keypoint.pt))
                        {
                            filteredKeypoints.push_back(keypoint);
                        }
                    }
                }
                else 
                {
                    filteredKeypoints = keypoints;
                }
                //// EOF STUDENT ASSIGNMENT
                if (bLimitKpts)
                {
                    int maxKeypoints = 50;

                    if (detectorType.compare("SHITOMASI") == 0)
                    { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                        filteredKeypoints.erase(filteredKeypoints.begin() + maxKeypoints, filteredKeypoints.end());
                    }
                    cv::KeyPointsFilter::retainBest(filteredKeypoints, maxKeypoints);
                    cout << "NOTE: keypoints have been limited!" << endl;
                }
                // push keypoints and descriptor for current frame to end of data buffer
                (dataBuffer.end() - 1)->keypoints = filteredKeypoints;
                cout << "#2 detect keypoints done" << endl;

                /* EXTRACT KEYPOINT DESCRIPTORS */
                //// STUDENT ASSIGNMENT
                //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
                //// -> BRIEF, ORB, FREAK, AKAZE, SIFT
                cv::Mat descriptors;
                try
                {
                    descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
                }
                catch (cv::Exception& e) {
                    cout << "(" << detectorType << ", " << descriptorType << ") are incompatible" << endl;
                    continue;
                }
                //// EOF STUDENT ASSIGNMENT
                // push descriptors for current frame to end of data buffer
                (dataBuffer.end() - 1)->descriptors = descriptors;
                cout << "#3 extract descriptors done" << endl;

                if (dataBuffer.size() > 1) // wait until at least two images have been processed
                {
                    /* MATCH KEYPOINT DESCRIPTORS */
                    vector<cv::DMatch> matches;
                    string matcherType = "MAT_BF";   // either of MAT_BF, MAT_FLANN
                    string descrType = "DES_BINARY"; // DES_BINARY, DES_HOG
                    string selectorType = "SEL_KNN"; // SEL_NN, SEL_KNN
                    //// STUDENT ASSIGNMENT
                    //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
                    //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp
                    try 
                    {
                        matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                        (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                                        matches, descrType, matcherType, selectorType);
                    }
                    catch (cv::Exception& e)
                    {
                        cout << "(" << detectorType << ", " << descriptorType << ") are incompatible" << endl;
                        continue;
                    }
                    //// EOF STUDENT ASSIGNMENT
                    // store matches in current data frame
                    (dataBuffer.end() - 1)->kptMatches = matches;
                    cout << "#4 match keypoint descriptors done." << endl;
                    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
                    totalMatches += matches.size();
                    totalTime += t;

                    if (bVis)
                    {
                        cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                        cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                        (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                        matches, matchImg,
                                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                                        vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                        string windowName = "Matching keypoints between two camera images";
                        cv::namedWindow(windowName, 7);
                        cv::imshow(windowName, matchImg);
                        cout << "Press key to continue to next image" << endl;
                        cv::waitKey(0); // wait for key to be pressed
                    }
                }

            } // eof loop over all images
            cout << "[MP.8/.9] (" << detectorType << ", " << descriptorType << ")"
                 << " with total " << totalMatches << " matches in average " << 1000 * totalTime / (1.0 * (imgEndIndex-imgStartIndex-1)) << " ms" << endl;

        }
    } 


    return 0;
}
