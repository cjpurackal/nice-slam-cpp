#include "Renderer.h"

Renderer::Renderer():
bound_(3,2)
{
	ray_batch_size = 500000;
	points_batch_size = 100000;
	lindisp = false;
	perturb = 0;
	N_samples = 32;
	N_surface = 16;
	N_importance = 0;

	scale = 1;
	occupancy = true;
	bound = torch::tensor({{-4.5, 3.82},{-1.5, 2.02}, {-3.0, 2.76}});

}

/*torch::Tensor*/ void Renderer::eval_points(torch::Tensor p, NICE decoders, c10::Dict<std::string, torch::Tensor> c, std::string stage)
{

	auto p_split = torch::split(p, points_batch_size);
	std::vector<torch::Tensor> rets;
	for (auto& pi: p_split)
	{
		auto mask_x = (pi.index({Slice(None), 0}) < bound.index({0,1}))  & (pi.index({Slice(None), 0}) < bound.index({0,0}));
		auto mask_y = (pi.index({Slice(None), 1}) < bound.index({1,1}))  & (pi.index({Slice(None), 1}) < bound.index({1,0}));
		auto mask_z = (pi.index({Slice(None), 2}) < bound.index({2,1}))  & (pi.index({Slice(None), 2}) < bound.index({2,0}));
		auto mask = mask_x & mask_y & mask_z;
		mask = ~mask;

		pi = pi.unsqueeze(0);
		auto ret = decoders.forward(pi, c, stage);
		// ret = ret.squeeze(0);

		// if ((ret.sizes().size() == 1) && (ret.sizes()[0] == 4))
		// 	ret = ret.unsqueeze(0);
		// ret.index_put_({mask, 3}, 100);
		// rets.push_back(ret);

	}
	// auto ret = torch::cat({rets}, 0);
	// return ret;
}

void Renderer::render_batch_ray(c10::Dict<std::string, torch::Tensor> c, NICE decoders, torch::Tensor rays_d, torch::Tensor rays_o, std::string stage, torch::Tensor gt_depth, torch::Tensor& rgb_map, torch::Tensor& depth_map, torch::Tensor& depth_var, torch::Tensor& weights)
{
	int n_samples = N_samples;
	int n_sufrace = N_surface;
	int n_importance = N_importance;

	int n_rays = rays_o.sizes()[0];

	torch::Tensor near;
	
	if (!gt_depth.defined())
	{
		N_surface = 0;
		near = torch::tensor({0.01});
	}
	else
	{
		gt_depth = gt_depth.reshape({-1,1});
		auto gt_depth_samples = gt_depth.tile({1, n_samples}); //using tile instead of repeat, might erorr idk
		near = gt_depth_samples*0.01;
	}

	torch::NoGradGuard noGrad;
	auto det_rays_o = rays_o.unsqueeze(-1); //rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
	auto det_rays_d = rays_d.unsqueeze(-1); //rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
	auto t = (bound.unsqueeze(0) - det_rays_o)/det_rays_d;
	auto far_bb_ = torch::min(std::get<0>(torch::max(t, 2)),1);
	auto far_bb = std::get<0>(far_bb_);
	far_bb = far_bb.unsqueeze(-1);
	far_bb += 0.01;

	torch::Tensor far, z_vals_surface;
	if (gt_depth.defined())
		far = torch::clamp(far_bb, torch::tensor({0}), std::get<0>(torch::max(gt_depth*1.2, 0)));
	else
		far = far_bb;

	if (N_surface > 0)
	{
		auto gt_none_zero_mask = gt_depth > 0;
		auto gt_none_zero = gt_depth.index({gt_none_zero_mask});
		gt_none_zero = gt_none_zero.unsqueeze(-1);
		auto gt_depth_surface = gt_none_zero.tile({1, N_surface});
		auto t_vals_surface = torch::linspace(0, 1, n_sufrace);
		auto z_vals_surface_depth_none_zero = 0.95 * gt_depth_surface * (1-t_vals_surface) + 1.05 * gt_depth_surface * (t_vals_surface);
		z_vals_surface = torch::zeros({gt_depth.sizes()[0], n_sufrace});
		gt_none_zero_mask = gt_none_zero_mask.squeeze(-1);
		z_vals_surface.index_put_({gt_none_zero_mask, Slice(None)}, z_vals_surface_depth_none_zero);
		auto near_surface = 0.001;
		auto far_surface = std::get<0>(torch::max(gt_depth, 0)); //asuming 0 dim
		auto z_vals_surface_depth_zero = near_surface * (1.-t_vals_surface) + far_surface * (t_vals_surface);
		// auto gt_none_zero_mask_sum = ~gt_none_zero_mask.sum();
		auto gt_none_zero_mask_sum = gt_depth.sizes()[0]-gt_none_zero_mask.sum();
		z_vals_surface_depth_zero = z_vals_surface_depth_zero.unsqueeze(0).tile({gt_none_zero_mask_sum.item<int>(), 1});
		z_vals_surface.index_put_({~gt_none_zero_mask, Slice(None)}, z_vals_surface_depth_zero); // doubtful to do, double check
	}

	auto t_vals = torch::linspace(0, 1, n_samples);
	
	torch::Tensor z_vals;
	if (!lindisp)
		z_vals = near * (1 - t_vals) + far * (t_vals); 
	else
		z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals));

	if (perturb > 0)
	{
		auto mids = .5 * (z_vals.index({"...", Slice(1,None)}) +z_vals.index({"...", Slice(None,-1)}));
		auto upper = torch::cat({mids, z_vals.index({"...", Slice(-1,None)})}, -1);
		auto lower = torch::cat({z_vals.index({"...", Slice(None,1)}), mids}, -1);
		auto t_rand = torch::rand(z_vals.sizes());	
		z_vals = lower + (upper - lower) * t_rand;
	}
	if (N_surface > 0)
		z_vals = std::get<0>(torch::sort(torch::cat({z_vals, z_vals_surface}, -1), -1));

	auto pts = rays_o.index({"...", None, Slice(None)}) + rays_d.index({"...", None, Slice(None)}) * z_vals.index({"...", Slice(None), None});
	auto pointsf = pts.reshape({-1,3});
	/* auto raw = */eval_points(pointsf, decoders, c, stage);
	// raw = raw.reshape({n_rays, n_samples+n_sufrace, -1});
	// raw2outputs_nerf_color(raw, z_vals, false, rays_d, rgb_map, depth_map, depth_var, weights);
}