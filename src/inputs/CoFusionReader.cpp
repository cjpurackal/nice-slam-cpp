//cjpurackal
//June 4 '22, 21:20:00
#include "inputs/CoFusionReader.h"

CoFusionReader::CoFusionReader(std::string inp)
{
	fptr = 850;
	input_folder = inp;
}

CoFusionReader::~CoFusionReader()
{

}

void CoFusionReader::getNext()
{
	std::string rgb_f = input_folder+"colour/Color0"+std::to_string(int(fptr/100))+std::to_string(int(fptr/10%10))+std::to_string(int(fptr%100%10))+".png";
	std::string depth_f = input_folder+"depth_original/Depth0"+std::to_string(int(fptr/100))+std::to_string(int(fptr/10%10))+std::to_string(int(fptr%100%10))+".png";

}