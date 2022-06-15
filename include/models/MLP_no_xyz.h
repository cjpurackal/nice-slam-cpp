#include <torch/torch.h>
#include "models/GaussianFFT.h"
#include "torchlib/utils.h"

struct MLP_no_xyz: torch::nn::Module
{
	MLP_no_xyz(std::string name, int dim, int c_dim, int hidden_size, int n_blocks, bool leak, std::string sample_mode, bool color, std::vector<int> skips, float grid_len, std::string pose_emb, bool concat_feat);
	torch::Tensor sample_grid_feature(torch::Tensor p, torch::Tensor grid_feature);
	torch::nn::Linear reset_parameters_dense(torch::nn::Linear layer, std::string activation);
	torch::Tensor forward(torch::Tensor p, std::map<std::string, torch::Tensor> c_grid);

	std::string name;
	bool color;
	int c_dim, n_blocks;
	std::vector<int> skips;
	float grid_len;
	torch::nn::ModuleList pts_linear;
	torch::nn::Linear output_linear;

};