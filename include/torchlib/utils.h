#include <torch/torch.h>
#include <torch/script.h>

using namespace torch::indexing;

void raySampler(int H0, int H1, int W0, int W1, int cx, int cy, int fx, int fy, cv::Mat rgb_, cv::Mat depth_, Eigen::Matrix4f c2w_)
{

	int n = 100;
	std::vector<torch::Tensor> tl;
	tl.push_back(torch::linspace(W0,W1-1,W1-W0,torch::TensorOptions().dtype(torch::kF32)/*.device(torch::kCUDA, 0)*/));
	tl.push_back(torch::linspace(H0,H1-1,H1-H0,torch::TensorOptions().dtype(torch::kF32)/*.device(torch::kCUDA, 0)*/));
	std::vector<torch::Tensor> t = torch::meshgrid(tl);
	torch::Tensor i = t[0];
	torch::Tensor j = t[1];

	i = i.reshape(-1);
	j = j.reshape(-1);

	torch::Tensor ind = torch::randint(i.sizes()[0], {n});
	ind = ind.clamp(0, i.sizes()[0]);
	ind = ind.to(torch::kLong);
	i = i.index({ind});
	j = j.index({ind});

	torch::Tensor rgb = torch::from_blob(rgb_.data, {rgb_.size().height, rgb_.size().width, 3});
	torch::Tensor depth = torch::from_blob(depth_.data, {depth_.size().height, depth_.size().width, 1});
	torch::Tensor c2w = torch::from_blob(c2w_.data(), {4,4});

	rgb = rgb.reshape(-1);
	depth = depth.reshape({-1, 3});

	rgb = rgb.index({ind});
	depth = depth.index({ind});

	auto i_t = (i-cx)/fx;
	auto j_t = (i-cy)/fy;

	auto dirs = torch::stack({i_t,j_t,torch::ones_like(i)});
	dirs = dirs.reshape({-1, 1, 3});

	auto tmp = dirs * c2w.index({Slice(None,3), Slice(None, 3)});
	auto rays_d = torch::sum(tmp, -1);
	auto rays_o = c2w.index({Slice(None, 3), -1}).expand(rays_d.sizes());

}