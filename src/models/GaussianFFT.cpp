#include "models/GaussianFFT.h"

GaussianFFT::GaussianFFT(int num_channels, int mapping_size, int scale)
{
	//assuming that learnable is true
	B = register_parameter("GFF", torch::randn({num_channels, mapping_size}) * scale);
}
GaussianFFT::GaussianFFT(){}

torch::Tensor GaussianFFT::forward(torch::Tensor x)
{
	x = x.squeeze(0);
	x = torch::matmul(x, B);
	return torch::sin(x);
}

