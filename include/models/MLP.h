#include <torch/torch.h>
#include "models/GaussianFFT.h"
#include "torchlib/utils.h"

struct MLP: torch::nn::Module
{
	MLP(std::string name, int dim, int c_dim, int hidden_size, int n_blocks, bool leak, std::string sample_mode, bool color, std::vector<int> skips, float grid_len, std::string pose_emb, bool concat_feat);
	torch::Tensor sample_grid_feature(torch::Tensor p, torch::Tensor c);
	torch::nn::Linear reset_parameters_dense(torch::nn::Linear layer, std::string activation) ;
	torch::Tensor forward(torch::Tensor p, std::map<std::string, torch::Tensor> c_grid);

	std::string name;
	bool color, no_grad_feature, concat_feat;
	int c_dim, n_blocks;
	std::vector<int> skips;
	float grid_len;
	torch::nn::ModuleList fc, pts_linear;
	torch::nn::Linear output_linear;
	GaussianFFT embedder;
	int embedding_size;

};