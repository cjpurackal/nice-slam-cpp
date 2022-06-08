//cjpurackal
//June 4 '22, 21:20:00

#ifndef COFUSIONREADER_H_
#define COFUSIONREADER_H_

#include <iostream>
#include <stdio.h>
#include <string>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Core>
#include "tinyexr.h"

class CoFusionReader
{
    public:
        CoFusionReader(std::string input_folder);
        virtual ~CoFusionReader();

        void getNext();
        void getBack();
        bool hasMore();
        void reset();
        
        std::string input_folder;
        cv::Mat depth, rgb;
        Eigen::Matrix4f c2w;

        int width, height;
        int fptr;
        float png_depth_scale;
};
#endif /* COFUSIONREADER_H_ */