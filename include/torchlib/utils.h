#include <torch/torch.h>
#include <torch/script.h>
#include <Eigen/LU>

using namespace torch::indexing;

void raySampler(int H0, int H1, int W0, int W1, int fx, int fy, int cx, int cy, cv::Mat rgb_, cv::Mat depth_, Eigen::Matrix4f c2w_, torch::Tensor& rays_d, torch::Tensor& rays_o, torch::Tensor& gt_color, torch::Tensor& gt_depth)
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

	depth = depth.reshape(-1);
	rgb = rgb.reshape({-1, 3});

	gt_color = rgb.index({ind});
	gt_depth = depth.index({ind});

	auto i_t = (i-cx)/fx;
	auto j_t = (i-cy)/fy;

	auto dirs = torch::stack({i_t,j_t,-torch::ones_like(i)}, -1);
	dirs = dirs.reshape({-1, 1, 3});

	rays_d = torch::sum(dirs * c2w.index({Slice(None,3), Slice(None, 3)}), -1);
	rays_o = c2w.index({Slice(None, 3), -1}).expand(rays_d.sizes());

}


void keyframe_selection_overlap(int H0, int H1, int W0, int W1, int fx, int fy, int cx, int cy, cv::Mat rgb_, cv::Mat depth_, Eigen::Matrix4f c2w_)
{

	int n_samples =16;

	torch::Tensor rays_o, rays_d, gt_color, gt_depth;
	raySampler(H0, H1, W0, W1, fx, fy, cx, cy, rgb_, depth_, c2w_, rays_d, rays_o, gt_color, gt_depth);

	gt_depth = gt_depth.reshape({-1, 1});
	gt_depth = gt_depth.repeat({1,n_samples});
	auto t_vals = torch::linspace(0, 1, n_samples);
	auto near = gt_depth * 0.8;
	auto far = gt_depth + 0.5;
	auto z_vals = near * (1 - t_vals) + far * (t_vals);
	auto pts = rays_o.index({"...", None, Slice(None)}) + rays_d.index({"...", None, Slice(None)}) * z_vals.index({"...", Slice(None), None});
	auto vertices = pts.reshape({-1, 3});

	std::vector<Eigen::Matrix4f> keyframe_vec;
	keyframe_vec.push_back(Eigen::Matrix4f::Identity(4,4));

	Eigen::Matrix3f k;
	k << fx, 0, cx,
		0, fy, cy,
		0, 0, 1;
	auto K = torch::from_blob(k.data(), {3, 3});

	for (int i = 0; i < keyframe_vec.size(); ++i)
	{
		Eigen::Matrix4f c2w = keyframe_vec[i];
		Eigen::Matrix4f w2c_ = c2w.inverse();
		auto w2c = torch::from_blob(w2c_.data(), {4,4});
		auto ones = torch::ones_like(vertices.index({Slice(None), 0})).reshape({-1, 1});
		auto homo_vertices = torch::cat({vertices, ones}, 1).reshape({-1, 4, 1});
		auto cam_cord_homo = torch::matmul(w2c, homo_vertices);
		auto cam_cord = cam_cord_homo.index({Slice(None), Slice(None, 3)});

		cam_cord = cam_cord.index({Slice(None), 0}) * -1;
		auto uv = torch::matmul(K, cam_cord);
		auto z = uv.index({Slice(None), Slice(-1, None)})+1e-5;
		uv = uv.index({Slice(None), Slice(None, 2)})/z;
		//uv to float conver here TODO
		edge = 20;

	}

}