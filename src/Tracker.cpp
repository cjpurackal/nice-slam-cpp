//cjpurackal
//June 11 '22, 16:54:00
#include "Tracker.h"

Tracker::Tracker(YAML::Node ns_config, YAML::Node cf_config, c10::Dict<std::string, torch::Tensor> c_dict):
renderer()
{
	// scale = 1;
	// coarse = false;
	// occupancy = false;
	// sync_method = "strict";
	// idx = 0;
	mapping_idx = 0;
	handle_dynamic = ns_config["tracking"]["handle_dynamic"].as<bool>();
	use_color_in_tracking = ns_config["tracking"]["use_color_in_tracking"].as<bool>();
	w_color_loss =  ns_config["tracking"]["w_color_loss"].as<float>();
	lr =  ns_config["tracking"]["lr"].as<float>();
	num_cam_iters =  ns_config["tracking"]["iters"].as<int>();
	tracking_pixels = ns_config["tracking"]["pixels"].as<int>();
	ignore_edge_w = ns_config["tracking"]["ignore_edge_W"].as<int>();
	ignore_edge_h = ns_config["tracking"]["ignore_edge_H"].as<int>();
	
	bound = torch::tensor({{-4.5, 3.82},{-1.5, 2.02}, {-3.0, 2.76}});
 
	H = cf_config["cam"]["H"].as<int>();
	W = cf_config["cam"]["W"].as<int>();
	fx = cf_config["cam"]["fx"].as<float>();
	fy = cf_config["cam"]["fy"].as<float>();
	cx = cf_config["cam"]["cx"].as<float>();
	cy = cf_config["cam"]["cy"].as<float>();

	current_min_loss = 10000000000;
	c = c_dict;
}

Tracker::~Tracker()
{

}

torch::Tensor Tracker::optimize_cam_in_batch(torch::Tensor cam_tensor, torch::Tensor gt_color, torch::Tensor gt_depth, int batch_size ,torch::optim::Adam& optimizer, NICE decoders)
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
	batch_gt_depth = batch_gt_depth.to(torch::Device(torch::kCUDA, 0));
	batch_gt_color = batch_gt_color.to(torch::Device(torch::kCUDA, 0));


	if (handle_dynamic)
	{
		auto tmp = torch::abs(batch_gt_depth-depth)/*/torch::sqrt(uncertainity+1e-10)*/;
		mask = (tmp < 10*tmp.median()) & (batch_gt_depth > 0);
	}
	else
		mask = batch_gt_depth > 0;

	auto loss = ((torch::abs(batch_gt_depth-depth))/torch::sqrt(uncertainity+1e-10)).index({mask}).sum();

	if (use_color_in_tracking)
	{
		auto color_loss = torch::abs(batch_gt_color - color);
		color_loss = color_loss.index({mask}).sum();
		loss = loss + w_color_loss*color_loss;
	}
	loss.requires_grad_(true);
	loss.backward();
	optimizer.step();
	optimizer.zero_grad();

	return loss;
}

void Tracker::update_para_from_mapping()
{
	// typename std::map<std::string, torch::Tensor>::iterator it = shared_c.begin();
	// if (mapping_idx != prev_mapping_idx)
	// {
	// 	for(std::pair<std::string, torch::Tensor> element : shared_c )
	// 	{
	// 		shared_c[element.first] = element.second; //change shared_c 
	// 	}
	// 	prev_mapping_idx = mapping_idx;
	// }
}

void Tracker::run(CoFusionReader cfreader, NICE decoders)
{
	while(cfreader.hasMore())
	{
		cfreader.getNext();
		auto gt_color = cfreader.rgb;
		auto gt_depth = cfreader.depth; 
		auto gt_color_t = torch::from_blob(gt_color.data, {cfreader.height, cfreader.width, 3});
		auto gt_depth_t = torch::from_blob(gt_depth.data, {cfreader.height, cfreader.width});

		auto c2w = cfreader.c2w;
		auto gt_c2w_t = torch::from_blob(c2w.data(), {4, 4});
		torch::Tensor camera_tensor, gt_camera_tensor;
		gt_camera_tensor = get_tensor_from_camera(gt_c2w_t, false);
		camera_tensor = get_tensor_from_camera(gt_c2w_t, false);


		std::vector<torch::Tensor> cam_para_list{camera_tensor};
		torch::optim::Adam optimizer(cam_para_list, torch::optim::AdamOptions(1e-2));

   		auto initial_loss_camera_tensor = torch::abs(gt_camera_tensor-camera_tensor);

   		num_cam_iters = 10;

   		for (int i = 0; i < num_cam_iters; ++i)
   		{
   			auto loss = optimize_cam_in_batch(camera_tensor, gt_color_t, gt_depth_t, tracking_pixels , optimizer, decoders);
   			std::cout<<"loss: "<<loss<<std::endl;
   		}
	}
}