#include <torch/torch.h>
#include "models/GaussianFFT.h"

struct MLP: torch::nn::Module
{
	MLP(std::string name, int dim, int c_dim, int hidden_size, int n_blocks, bool leak, std::string sample_mode, bool color, std::vector<int> skips, float grid_len, std::string pose_emb, bool concat_feat):
	name(name),
	color(color),
	no_grad_feature(no_grad_feature),
	c_dim(c_dim),
	grid_len(grid_len),
	concat_feat(concat_feat),
	n_blocks(n_blocks),
	skips(skips)
	{
		//assuming number of blocks to be 5 for NICE
		fc = torch::nn::ModuleList(
			torch::nn::Linear(c_dim,hidden_size),
			torch::nn::Linear(c_dim,hidden_size),
			torch::nn::Linear(c_dim,hidden_size),
			torch::nn::Linear(c_dim,hidden_size),
			torch::nn::Linear(c_dim,hidden_size)
			);
		embedder = GaussianFFT(dim,93,25);
		// pts_linear = torch::nn::ModuleList(
		// 	torch::nn::Linear()
		// 	);

	}

	std::string name;
	bool color, no_grad_feature, concat_feat;
	int c_dim, n_blocks;
	std::vector<int> skips;
	float grid_len;
	torch::nn::ModuleList fc, pts_linear;
	GaussianFFT embedder;


};