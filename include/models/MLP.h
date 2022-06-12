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
		//assuming number of blocks to be 5 for NICE and embedder to be fourier
		fc = torch::nn::ModuleList(
			torch::nn::Linear(c_dim,hidden_size),
			torch::nn::Linear(c_dim,hidden_size),
			torch::nn::Linear(c_dim,hidden_size),
			torch::nn::Linear(c_dim,hidden_size),
			torch::nn::Linear(c_dim,hidden_size)
			);
		embedding_size = 93;
		embedder = GaussianFFT(dim,embedding_size,25);

		torch::nn::Linear pts_linear_0(embedding_size, hidden_size);
		pts_linear_0 = reset_parameters_dense(pts_linear_0, "relu");
		torch::nn::Linear pts_linear_1(hidden_size, hidden_size);
		pts_linear_1 = reset_parameters_dense(pts_linear_1, "relu");
		torch::nn::Linear pts_linear_2(hidden_size, hidden_size);
		pts_linear_2 = reset_parameters_dense(pts_linear_2, "relu");
		//skip connection
		torch::nn::Linear pts_linear_3(hidden_size+embedding_size, hidden_size);
		pts_linear_3 = reset_parameters_dense(pts_linear_3, "relu");		
		torch::nn::Linear pts_linear_4(hidden_size, hidden_size);
		pts_linear_4 = reset_parameters_dense(pts_linear_4, "relu");
		pts_linear = torch::nn::ModuleList(
			pts_linear_0,
			pts_linear_1,
			pts_linear_2,
			pts_linear_3,
			pts_linear_4
			);

		torch::nn::Linear output_linear(hidden_size,4); //tmp initialization

		if (color)
			output_linear = reset_parameters_dense(torch::nn::Linear(hidden_size,4), "linear");
		else
			output_linear = reset_parameters_dense(torch::nn::Linear(hidden_size,1), "linear");


	}

	torch::nn::Linear reset_parameters_dense(torch::nn::Linear layer, std::string activation) 
	{
		torch::NoGradGuard noGrad;
		if (activation == "relu")
			torch::nn::init::xavier_uniform_(layer->weight, torch::nn::init::calculate_gain(torch::enumtype::kReLU()));
		else if (activation == "linear")
			torch::nn::init::xavier_uniform_(layer->weight, torch::nn::init::calculate_gain(torch::enumtype::kReLU()));
		torch::nn::init::zeros_(layer->bias);
		return layer;
	}


	std::string name;
	bool color, no_grad_feature, concat_feat;
	int c_dim, n_blocks;
	std::vector<int> skips;
	float grid_len;
	torch::nn::ModuleList fc, pts_linear;
	GaussianFFT embedder;
	int embedding_size;

};