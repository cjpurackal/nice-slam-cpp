#include <torch/torch.h>
#include "models/MLP.h"

struct NICE: torch::nn::Module
{
	NICE(int dim, int c_dim, int hidden_size, float coarse_grid_len, float middle_grid_len, float fine_grid_len, float color_grid_len, bool coarse, std::string pose_emb);
	torch::Tensor forward(torch::Tensor p, c10::Dict<std::string, torch::Tensor> c_grid, std::string stage);

	MLP middle_decoder, fine_decoder, color_decoder;
	MLP_no_xyz coarse_decoder;
	torch::jit::script::Module cmodule, mmodule, fmodule;
};