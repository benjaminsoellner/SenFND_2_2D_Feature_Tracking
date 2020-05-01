#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>


#include "dataStructures.h"
#include "structIO.hpp"

using namespace std;

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &keypointsPrev, std::vector<cv::KeyPoint> &keypointsCurr,
                      std::vector<cv::DMatch> keypointMatches, double frameRate, double &TTC)
{
    // compute distance ratios between all matched keypoints
    vector<double> distanceRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = keypointMatches.begin(); it1 != keypointMatches.end() - 1; ++it1)
    {
        // outer keypoint loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = keypointsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = keypointsPrev.at(it1->queryIdx);

        for (auto it2 = keypointMatches.begin() + 1; it2 != keypointMatches.end(); ++it2)
        {
            // inner keypoint loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = keypointsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = keypointsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distanceCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distancePrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distancePrev > std::numeric_limits<double>::epsilon() && distanceCurr >= minDist)
            {
                // avoid division by zero

                double distRatio = distanceCurr / distancePrev;
                distanceRatios.push_back(distRatio);
            }

            // /inner keypoint loop
        } 

        // /outer keypoint loop
    }

    // only continue if list of distance ratios is not empty
    if (distanceRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }


    // STUDENT TASK (replacement for meanDistRatio)
    // compute median distance. ratio to remove outlier influence
    std::sort(distanceRatios.begin(), distanceRatios.end());
    long medianIndex = floor(distanceRatios.size() / 2.0);
    double medianDistanceRatio;
    if (distanceRatios.size() % 2 == 0)
    {
        medianDistanceRatio = (distanceRatios[medianIndex - 1] + distanceRatios[medianIndex]) / 2.0;
    }
    else 
    {
        medianDistanceRatio = distanceRatios[medianIndex];
    }

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medianDistanceRatio);
    // EOF STUDENT TASK
}

int main()
{
    vector<cv::KeyPoint> keypointsSource, keypointsRef;
    readKeypoints("../dat/C23A5_KptsSource_AKAZE.dat", keypointsSource); // readKeypoints("./dat/C23A5_KptsSource_SHI-BRISK.dat"
    readKeypoints("../dat/C23A5_KptsRef_AKAZE.dat", keypointsRef); // readKeypoints("./dat/C23A5_KptsRef_SHI-BRISK.dat"

    vector<cv::DMatch> matches;
    readKptMatches("../dat/C23A5_KptMatches_AKAZE.dat", matches); // readKptMatches("./dat/C23A5_KptMatches_SHI-BRISK.dat", matches);
    double ttc; 
    computeTTCCamera(keypointsSource, keypointsRef, matches, 10.0, ttc);
    cout << "ttc = " << ttc << "s" << endl;
}