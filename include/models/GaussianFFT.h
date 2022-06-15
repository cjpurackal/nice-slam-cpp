#include <iostream>
#include <torch/torch.h>

struct GaussianFFT: torch::nn::Module
{
	GaussianFFT(int num_channels, int mapping_size, int scale);
	GaussianFFT();
	torch::Tensor forward(torch::Tensor x);
	
	torch::Tensor B;
};