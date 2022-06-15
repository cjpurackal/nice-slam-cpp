#include "models/MLP.h"

MLP::MLP(std::string name, int dim, int c_dim, int hidden_size, int n_blocks, bool leak, std::string sample_mode, bool color, std::vector<int> skips, float grid_len, std::string pose_emb, bool concat_feat):
name(name),
color(color),
no_grad_feature(no_grad_feature),
c_dim(c_dim),
grid_len(grid_len),
concat_feat(concat_feat),
n_blocks(n_blocks),
skips(skips),
output_linear(register_module("linear", torch::nn::Linear(hidden_size, 4)))
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

	if (color)
		output_linear = reset_parameters_dense(torch::nn::Linear(hidden_size,4), "linear");
	else
		output_linear = reset_parameters_dense(torch::nn::Linear(hidden_size,1), "linear");


}

torch::Tensor MLP::sample_grid_feature(torch::Tensor p, torch::Tensor c)
{
	Eigen::MatrixXf bound(3,2);
	bound<<-4.5, 3.82,
		   -1.5, 2.02,
		   -3, 2.76 ;
	auto bound_t = torch::from_blob(bound.data(), {3, 2});
	normalize_3d_coordinate(p, bound_t);
	auto p_nor = p.unsqueeze(0);
	auto vgrid = p_nor.index({Slice(None), Slice(None), None, None});
	F::grid_sample(c, vgrid, F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kBorder).align_corners(true)).squeeze(-1).squeeze(-1);
	return c;
}

torch::nn::Linear MLP::reset_parameters_dense(torch::nn::Linear layer, std::string activation) 
{
	torch::NoGradGuard noGrad;
	if (activation == "relu")
		torch::nn::init::xavier_uniform_(layer->weight, torch::nn::init::calculate_gain(torch::enumtype::kReLU()));
	else if (activation == "linear")
		torch::nn::init::xavier_uniform_(layer->weight, torch::nn::init::calculate_gain(torch::enumtype::kReLU()));
	torch::nn::init::zeros_(layer->bias);
	return layer;
}

torch::Tensor MLP::forward(torch::Tensor p, std::map<std::string, torch::Tensor> c_grid)
{
	auto c = sample_grid_feature(p, c_grid["grid_"+name]).transpose(1, 2).squeeze(0);
	if (concat_feat)
	torch::NoGradGuard noGrad;
	auto c_middle = sample_grid_feature(p, c_grid["grid_middle"]).transpose(1, 2).squeeze(0);
	c = torch::cat({c, c_middle}, 1);

	torch::Tensor embedded_pts = embedder.forward(p);
	auto h = embedded_pts;

	int i = 0;
	for (const auto &module : *pts_linear) 
	{
		h = module->as<torch::nn::Linear>()->forward(embedded_pts);
		h = F::relu(h);
		h = h + fc[i]->as<torch::nn::Linear>()->forward(c);
		if (std::find(skips.begin(), skips.end(), i) != skips.end())
			h = torch::cat({embedded_pts, h}, -1);
	}
	auto out = output_linear(h);
	if (!color)
		out = out.squeeze(-1);
	return out;
}