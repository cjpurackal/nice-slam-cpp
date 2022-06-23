//cjpurackal
//June 11 '22, 16:54:00
#include "Tracker.h"


Tracker::Tracker(YAML::Node cf_config):
renderer()
{
	// scale = 1;
	// coarse = false;
	// occupancy = false;
	// sync_method = "strict";

	// idx = 0;
	mapping_idx = 0;
	handle_dynamic = cf_config["tracking"]["handle_dynamic"].as<bool>();
	use_color_in_tracking = cf_config["tracking"]["use_color_in_tracking"].as<bool>();
	w_color_loss =  cf_config["tracking"]["w_color_loss"].as<float>();
	Eigen::MatrixXf bound_;
	bound_<< -4.5, 3.82,
		-1.5, 2.02,
		-3, 2.76 ;
	bound = torch::from_blob(bound_.data(), {3, 2});
	H=0;
	W=0;

}
Tracker::~Tracker()
{

}

torch::Tensor Tracker::optimize_cam_in_batch(torch::Tensor cam_tensor, torch::Tensor gt_color, torch::Tensor gt_depth, int batch_size ,torch::optim::Adam optimizer, NICE decoders)
{
	optimizer.zero_grad();
	auto c2w = get_camera_from_tensor(cam_tensor);
	torch::Tensor batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color;
	get_samples(ignore_edge_h, H-ignore_edge_h, ignore_edge_w, W-ignore_edge_w, batch_size, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color);
	torch::NoGradGuard noGrad;
	auto det_rays_o = batch_rays_o.unsqueeze(-1);
	auto det_rays_d = batch_rays_d.unsqueeze(-1);

	auto t_ = (bound.unsqueeze(0)-det_rays_o)/det_rays_d;
	auto t = std::get<0>(torch::min(std::get<0>(torch::max(t_, 2)),1));
	auto inside_mask = t>=batch_gt_depth;
	batch_rays_d = batch_rays_d.index({inside_mask});
	batch_rays_o = batch_rays_o.index({inside_mask});
    batch_gt_depth = batch_gt_depth.index({inside_mask});
    batch_gt_color = batch_gt_color.index({inside_mask});
	
	torch::Tensor color, depth, uncertainity, weights;
	renderer.render_batch_ray(c, decoders, batch_rays_d, batch_rays_o, "color", batch_gt_depth, color, depth, uncertainity, weights);
	torch::Tensor mask;

	if (handle_dynamic)
	{
		auto tmp = torch::abs(batch_gt_depth-depth)/torch::sqrt(uncertainity+1e-10);
		mask = (tmp < 10*tmp.median()) & (batch_gt_depth > 0);
	}
	else
		mask = batch_gt_depth > 0;

	auto loss = (torch::abs(batch_gt_depth-depth))/torch::sqrt(uncertainity+1e-10);
	loss = loss.index({mask}).sum();

	if (use_color_in_tracking)
	{
		auto color_loss = torch::abs(batch_gt_color - color);
		color_loss = color_loss.index({mask}).sum();
		loss = loss + w_color_loss*color_loss;
	}

	loss.backward();
	optimizer.step();
	optimizer.zero_grad();

	return loss;

}

void Tracker::update_para_from_mapping()
{
	typename std::map<std::string, torch::Tensor>::iterator it = shared_c.begin();
	if (mapping_idx != prev_mapping_idx)
	{
		for(std::pair<std::string, torch::Tensor> element : shared_c )
		{
			shared_c[element.first] = element.second; //change shared_c 
		}
		prev_mapping_idx = mapping_idx;
	}
}

void Tracker::run(CoFusionReader cfreader)
{


}

