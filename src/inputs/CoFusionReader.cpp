//cjpurackal
//June 4 '22, 21:20:00
#define TINYEXR_IMPLEMENTATION
#include "inputs/CoFusionReader.h"

CoFusionReader::CoFusionReader(std::string inp)
{
	fptr = 1;
	input_folder = inp;
	width=640;
	height=480;
	png_depth_scale = 6553.5;

	c2w = Eigen::MatrixXf::Identity(4,4);
}

CoFusionReader::~CoFusionReader()
{

}

bool CoFusionReader::hasMore()
{
	if (fptr > 849)
		return false;
	else
		return true;
}

void CoFusionReader::getNext()
{
	if (hasMore())
	{
		std::string rgb_f = input_folder+"colour/Color0"+std::to_string(int(fptr/100))+std::to_string(int(fptr/10%10))+std::to_string(int(fptr%100%10))+".png";
		std::string depth_f = input_folder+"depth_noise/Depth0"+std::to_string(int(fptr/100))+std::to_string(int(fptr/10%10))+std::to_string(int(fptr%100%10))+".exr";
		const char* err = NULL;
		const char* depth_f_ = depth_f.c_str();
		float* out; // width * height * RGBA
		LoadEXR(&out, &width, &height, depth_f_, &err);
		depth = cv::Mat(height, width, CV_32FC1, out);

		rgb = cv::imread(rgb_f, cv::IMREAD_UNCHANGED);
	    rgb.convertTo(rgb, CV_32FC3,  1.0/ 255.0);
	    depth.convertTo(depth, CV_32FC1,  1.0/ png_depth_scale);

	    fptr++;
	}
	else
	{
		std::cout<<fptr<<"! fptr size exceeded."<<std::endl;
	}
}

void CoFusionReader::getBack()
{

}

void CoFusionReader::reset()
{
	fptr = 1;
}