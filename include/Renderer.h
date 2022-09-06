#ifndef RENDERER
#define RENDERER

#include <Eigen/Core>
#include <torch/torch.h>
#include "models/NICE.h"

class Renderer
{
	public:
		Renderer();
		torch::Tensor eval_points(torch::Tensor p, NICE decoders, c10::Dict<std::string, torch::Tensor> c, std::string stage);
		void render_batch_ray(c10::Dict<std::string, torch::Tensor> c, NICE decoders, torch::Tensor rays_d, torch::Tensor rays_o, std::string stage, torch::Tensor gt_depth, torch::Tensor& rgb_map, torch::Tensor& depth_map, torch::Tensor& depth_var, torch::Tensor& weights);

	private:
		int points_batch_size, ray_batch_size;
		bool lindisp, occupancy;
		float perturb;
		int N_samples, N_surface, N_importance;
		int scale;
		torch::Tensor bound;
		int H,W;
		float fx, fy, cx, cy;
};

#endif