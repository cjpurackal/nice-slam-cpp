//cjpurackal
//June 10 '22, 9:00:00

#include "Mapper.h"

Mapper::Mapper(YAML::Node ns_config, YAML::Node cf_config, bool cmapr):
renderer()
{
	ns_cfg = ns_config;
	cf_cfg = cf_config;
	color_refine = ns_cfg["mapping"]["color_refine"].as<bool>();
	mapping_window_size = ns_cfg["mapping"]["mapping_window_size"].as<int>();
	middle_iter_ratio = ns_cfg["mapping"]["middle_iter_ratio"].as<float>();
	fine_iter_ratio = ns_cfg["mapping"]["fine_iter_ratio"].as<float>();
	fix_color = ns_cfg["mapping"]["fix_color"].as<bool>();
	fix_fine = ns_cfg["mapping"]["fix_fine"].as<bool>();
	keyframe_selection_method = ns_cfg["mapping"]["keyframe_selection_method"].as<std::string>();
	frustum_feature_selection = ns_cfg["mapping"]["frustum_feature_selection"].as<bool>();
	keyframe_every = ns_cfg["mapping"]["keyframe_every"].as<int>();
	BA = false;
	coarse_mapper = cmapr;
	H = cf_config["cam"]["H"].as<int>();
	W = cf_config["cam"]["W"].as<int>();
	fx = cf_config["cam"]["fx"].as<float>();
	fy = cf_config["cam"]["fy"].as<float>();
	cx = cf_config["cam"]["cx"].as<float>();
	cy = cf_config["cam"]["cy"].as<float>();
	mapping_pixels = cf_config["mapping"]["pixels"].as<int>();
	bound = torch::tensor({{-4.5, 3.82},{-1.5, 2.02}, {-3.0, 2.76}});
	num_joint_iters = ns_cfg["mapping"]["iters"].as<int>();
	lr_factor = ns_cfg["mapping"]["lr_first_factor"].as<float>();
	BA_cam_lr = ns_cfg["mapping"]["BA_cam_lr"].as<float>();
	w_color_loss =  ns_cfg["tracking"]["w_color_loss"].as<float>();


}
Mapper::~Mapper()
{

}

void Mapper::get_mask_from_c2w(cv::Mat depth_mat, torch::Tensor c2w, torch::Tensor val_shape, std::string key, torch::Tensor& mask)
{
	typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_rm; // same as MatrixXf, but with row-major memory layout
	Eigen::Map<MatrixXf_rm> c2w_e(c2w.data_ptr<float>(), c2w.size(0), c2w.size(1));

	std::vector<torch::Tensor> tl;
	tl.push_back(torch::linspace(bound[0][0].item<int>(),bound[0][1].item<int>(),val_shape[2].item<int>()));
	tl.push_back(torch::linspace(bound[1][0].item<int>(),bound[1][1].item<int>(),val_shape[1].item<int>()));
	tl.push_back(torch::linspace(bound[2][0].item<int>(),bound[2][1].item<int>(),val_shape[0].item<int>()));
	std::vector<torch::Tensor> t = torch::meshgrid(tl);
	auto points = torch::stack({t[0], t[1], t[2]}, -1).reshape({-1, 3});

	if (key == "grid_coarse")
	{
		mask = torch::ones({val_shape[2].item<int>(),val_shape[1].item<int>(),val_shape[0].item<int>()});
		return;
		
	}

	Eigen::Matrix4f w2c_ = c2w_e.inverse();
	auto w2c = torch::from_blob(w2c_.data(), {4,4});
	auto ones = torch::ones_like(points.index({Slice(None), 0})).reshape({-1, 1});
	auto homo_vertices = torch::cat({points, ones}, 1).reshape({-1, 4, 1});
	auto cam_cord_homo = torch::matmul(w2c, homo_vertices);
	auto cam_cord = cam_cord_homo.index({Slice(None), Slice(None, 3)});

	Eigen::Matrix3f k;
	k << fx, 0, cx,
		0, fy, cy,
		0, 0, 1;
	auto K = torch::from_blob(k.data(), {3, 3});
	cam_cord.index_put_({Slice(None), 0}, cam_cord.index({Slice(None), 0})*-1);
	auto uv = torch::matmul(K, cam_cord);
	auto z = uv.index({Slice(None), Slice(-1, None)})+1e-5;
	uv = uv.index({Slice(None), Slice(None, 2)})/z;
	uv = uv.to(torch::kFloat32);

	int remap_chunk = int(3e4);
	std::vector<torch::Tensor> depths_vec;
	for (int i=0; i<uv.sizes()[0]; i+=remap_chunk)
	{
		auto map_x = uv.index({Slice(i, i+remap_chunk), 0});
		auto map_y = uv.index({Slice(i, i+remap_chunk), 1});

		cv::Mat mapx_mat = cv::Mat::eye(1,map_x.sizes()[0],CV_32F);
		cv::Mat mapy_mat = cv::Mat::eye(1,map_x.sizes()[0],CV_32F);

		std::memcpy(map_x.data_ptr(), mapx_mat.data, sizeof(float)*map_x.numel());
		std::memcpy(map_y.data_ptr(), mapy_mat.data, sizeof(float)*map_y.numel());

		cv::Mat depth_mat_(map_x.sizes()[0], 1, CV_32FC1);
		cv::remap(depth_mat, depth_mat_, mapx_mat, mapy_mat, cv::INTER_LINEAR);

		torch::Tensor depth_t=torch::zeros({map_x.sizes()[0]});
		std::memcpy(depth_mat_.data, depth_t.data_ptr(), sizeof(float)*map_x.sizes()[0]);
		depths_vec.push_back(depth_t);
	}
	auto depths = torch::cat(depths_vec, 0);

	int edge = 0;
	auto mask_ = (uv.index({Slice(None), 0}) < W-edge) * (uv.index({Slice(None), 0}) > edge) * (uv.index({Slice(None), 1}) < H-edge) * (uv.index({Slice(None), 1}) > edge);

	auto zero_mask = (depths == 0);
	depths.index_put_({zero_mask}, torch::max(depths).item<float>());
	// std::cout<<depths.sizes()<<" z sizes :"<<z.sizes()<<std::endl;
	// depth test

	auto mask_tmp = mask_ & (0 <= -1 * z.index({Slice(None), Slice(None), 0})); 
	auto mask_tmp2 = z.index({Slice(None), 0, 0}) <= depths+0.5;
	mask_ = mask_tmp & mask_tmp2.unsqueeze(-1);
	mask_ = mask_.reshape({-1});

	// # add feature grid near cam center
	auto ray_o_ = c2w_e.topRightCorner(3,1);
	auto ray_o = torch::from_blob(ray_o_.data(), {3,1}).unsqueeze(0);
	ray_o = ray_o.squeeze(-1);



	auto dist_ = points-ray_o;
	auto dist = torch::sum(dist_*dist_, 1);
	auto mask2 = dist < 0.5*0.5;

	mask = mask_ | mask2;

	points = points.index({mask});
	mask = mask.reshape({val_shape[2].item<int>(), val_shape[1].item<int>(), val_shape[0].item<int>()});

}

void Mapper::keyframe_selection_overlap(torch::Tensor gt_color_, torch::Tensor gt_depth_, torch::Tensor c2w, std::vector<KeyFrame> keyframe_vector_, int k_overlap, std::vector<int>& selected_kf)
{
	torch::Tensor rays_o, rays_d, gt_depth, gt_color;
	typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_rm; // same as MatrixXf, but with row-major memory layout
	int N_samples=16, pixels=100;
	get_samples(0, H, 0, W, pixels, H, W, fx, fy, cx, cy, c2w, gt_depth_, gt_color_, rays_o, rays_d, gt_depth, gt_color);
	gt_depth = gt_depth.reshape({-1,1});
	gt_depth = gt_depth.tile({1, N_samples});
	auto t_vals = torch::linspace(0, 1, N_samples).to(torch::Device(torch::kCUDA, 0));
	auto near = gt_depth*0.8;
	auto far = gt_depth+0.5;

	auto z_vals = near * (1.-t_vals) + far * (t_vals);
	auto pts = rays_o.index({"...", None, Slice(None)}) + rays_d.index({"...", None, Slice(None)}) * z_vals.index({"...", Slice(None), None});
	auto vertices = pts.reshape({-1, 3});

	Eigen::Matrix3f k;
	k << fx, 0, cx,
		0, fy, cy,
		0, 0, 1;
	auto K = torch::from_blob(k.data(), {3, 3});
	std::map<int, float> list_keyframe;

	for (int i=0; i<keyframe_vector_.size(); i++)
	{
		c2w = keyframe_vector_[i].est_c2w;
		Eigen::Map<MatrixXf_rm> c2w_e(c2w.data_ptr<float>(), c2w.size(0), c2w.size(1));
		Eigen::Matrix4f w2c_e = c2w_e.inverse();
		auto w2c = torch::from_blob(w2c_e.data(), {4,4});
		auto ones = torch::ones_like(vertices.index({Slice(None), 0})).reshape({-1, 1});
		auto homo_vertices = torch::cat({vertices, ones}, 1).reshape({-1, 4, 1});
		auto cam_cord_homo = torch::matmul(w2c, homo_vertices);
		auto cam_cord = cam_cord_homo.index({Slice(None), Slice(None, 3)});

		cam_cord.index({Slice(None), 0}) = cam_cord.index({Slice(None), 0}) * -1;
		auto uv = torch::matmul(K, cam_cord);
		auto z = uv.index({Slice(None), Slice(-1, None)})+1e-5;
		uv = uv.index({Slice(None), Slice(None, 2)})/z;
		uv = uv.to(torch::kFloat32);
		int edge = 20;
		auto mask = (uv.index({Slice(None), 0}) < W-edge) * (uv.index({Slice(None), 0}) > edge) * (uv.index({Slice(None), 1}) < H-edge) * (uv.index({Slice(None), 1}) > edge);
		mask = mask & (z.index({Slice(None), Slice(None), 0}) < 0);
		mask = mask.reshape(-1);
		auto percentage_inside = mask.sum()/uv.sizes()[0];
		float p_calc = percentage_inside.item<float>();	
		if (p_calc > 0)
			list_keyframe.insert({i, p_calc});

	}
	//sort the list_keyframe acc to perctage_inside
	std::vector<pair> sorted_kf;
	std::copy(list_keyframe.begin(),
			list_keyframe.end(),
			std::back_inserter<std::vector<pair>>(sorted_kf));
	std::sort(sorted_kf.begin(), sorted_kf.end(),
			[](const pair &l, const pair &r)
			{
			  return l.second > r.second;
			});
	for (auto ele : sorted_kf)
		selected_kf.push_back(ele.first);

	if ((selected_kf.size()-k_overlap) > 0)
		selected_kf.assign(selected_kf.begin(), selected_kf.begin()+k_overlap);
}

void Mapper::optimize_map(int num_joint_iters_, c10::Dict<std::string, torch::Tensor>& c, torch::Tensor cur_gt_color, torch::Tensor cur_gt_depth, torch::Tensor gt_cur_c2w,  torch::Tensor& cur_c2w, NICE& decoders)
{
	std::vector<int> optimize_frame;
	torch::Tensor bottom = torch::tensor({0,0,0,1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
	int num;
	if (keyframe_vector.size() != 0)
	{
		//assuming selection metho to be overlap
		num = mapping_window_size-2;
		keyframe_selection_overlap(cur_gt_color, cur_gt_depth, cur_c2w, keyframe_vector/*[:-1]*/, num, optimize_frame);
	}

	int oldest_frame = -1;
	if (keyframe_lvector.size() > 0)
	{
		optimize_frame.push_back(keyframe_lvector.size()-1);
		oldest_frame = *min_element(optimize_frame.begin(), optimize_frame.end());
	}	
	optimize_frame.push_back(-1);

	std::vector<torch::Tensor> coarse_grid_para, middle_grid_para, fine_grid_para, color_grid_para;
	std::vector<torch::Tensor> decoders_para_list, camera_tensor_list;
	torch::Tensor coarse_val, coarse_val_grad, middle_val, middle_val_grad, fine_val, fine_val_grad, color_val, color_val_grad; 
	torch::Tensor coarse_mask, middle_mask, fine_mask, color_mask;
	//keyinfo saving to do
	int pixs_per_image = int(mapping_pixels/optimize_frame.size());
	std::map<std::string, torch::Tensor> masked_c_grad;
	torch::Tensor mask_c2w;
	if (frustum_feature_selection)
		mask_c2w = cur_c2w;
	if (!frustum_feature_selection)
	{
		coarse_val = c.at("grid_coarse");
		coarse_val.to(torch::Device(torch::kCUDA, 0));
		coarse_val.requires_grad_(true);
		c.insert(std::string("grid_coarse"), coarse_val);
		coarse_grid_para.push_back(coarse_val);

		middle_val = c.at("grid_middle");
		middle_val.to(torch::Device(torch::kCUDA, 0));
		middle_val.requires_grad_(true);
		c.insert(std::string("grid_middle"), middle_val);
		middle_grid_para.push_back(middle_val);

		fine_val = c.at("grid_fine");
		fine_val.to(torch::Device(torch::kCUDA, 0));
		fine_val.requires_grad_(true);
		c.insert(std::string("grid_fine"), fine_val);
		fine_grid_para.push_back(fine_val);

		color_val = c.at("grid_color");
		color_val.to(torch::Device(torch::kCUDA, 0));
		color_val.requires_grad_(true);
		c.insert(std::string("grid_color"), color_val);
		color_grid_para.push_back(color_val);
	}
	else
	{
		cv::Mat depth_mat(cur_gt_depth.sizes()[0], cur_gt_depth.sizes()[1], CV_32FC1);
		std::memcpy(cur_gt_depth.data_ptr(), depth_mat.data, sizeof(float)*cur_gt_depth.numel());

		coarse_val = c.at("grid_coarse");
		get_mask_from_c2w(depth_mat, mask_c2w, torch::tensor({coarse_val.sizes()[2],coarse_val.sizes()[3], coarse_val.sizes()[4]}), "grid_coarse", coarse_mask);
		coarse_mask = coarse_mask.permute({2,1,0}).unsqueeze(0).unsqueeze(0).tile({1,coarse_val.sizes()[1], 1, 1, 1}); //using tile instead of repeat
		coarse_mask = coarse_mask.to(torch::kBool);
		coarse_val.to(torch::Device(torch::kCUDA, 0));
		coarse_val_grad = coarse_val.index({coarse_mask});
		coarse_grid_para.push_back(coarse_val_grad);

		middle_val = c.at("grid_middle");
		get_mask_from_c2w(depth_mat, mask_c2w, torch::tensor({middle_val.sizes()[2],middle_val.sizes()[3], middle_val.sizes()[4]}), "grid_middle", middle_mask);
		middle_mask = middle_mask.permute({2,1,0}).unsqueeze(0).unsqueeze(0).tile({1,middle_val.sizes()[1], 1, 1, 1}); //using tile instead of repeat
		middle_mask = middle_mask.to(torch::kBool);
		middle_val.to(torch::Device(torch::kCUDA, 0));
		middle_val_grad = middle_val.index({middle_mask});
		middle_grid_para.push_back(middle_val_grad);

		fine_val = c.at("grid_fine");
		get_mask_from_c2w(depth_mat, mask_c2w, torch::tensor({fine_val.sizes()[2],fine_val.sizes()[3], fine_val.sizes()[4]}), "grid_fine", fine_mask);
		fine_mask = fine_mask.permute({2,1,0}).unsqueeze(0).unsqueeze(0).tile({1,fine_val.sizes()[1], 1, 1, 1}); //using tile instead of repeat
		fine_mask = fine_mask.to(torch::kBool);
		fine_val.to(torch::Device(torch::kCUDA, 0));
		fine_val_grad = fine_val.index({fine_mask});
		fine_grid_para.push_back(fine_val_grad);

		color_val = c.at("grid_color");
		get_mask_from_c2w(depth_mat, mask_c2w, torch::tensor({color_val.sizes()[2],color_val.sizes()[3], color_val.sizes()[4]}), "grid_color", color_mask);
		color_mask = color_mask.permute({2,1,0}).unsqueeze(0).unsqueeze(0).tile({1,color_val.sizes()[1], 1, 1, 1}); //using tile instead of repeat
		color_mask = color_mask.to(torch::kBool);
		color_val.to(torch::Device(torch::kCUDA, 0));
		color_val_grad = color_val.index({color_mask});
		color_grid_para.push_back(color_val_grad);
	}

	if(!fix_fine)
	{
		auto fd_params = decoders.fine_decoder.parameters();
		decoders_para_list.insert(decoders_para_list.end(), fd_params.begin(), fd_params.end()); //double check this appending
	}
	if(!fix_color)
	{
		auto cd_params = decoders.color_decoder.parameters();
		decoders_para_list.insert(decoders_para_list.end(), cd_params.begin(), cd_params.end());			
	}
	
	torch::Tensor c2w, gt_c2w, camera_tensor;
	
	if (BA)
	{
		for (auto& frame : optimize_frame)	
		{
			if (frame != oldest_frame)
			{
				if (frame != -1)
				{
					c2w = keyframe_vector[frame].est_c2w;
					gt_c2w = keyframe_vector[frame].gt_c2w;
				}
				else
				{
					c2w = cur_c2w;
					gt_c2w = gt_cur_c2w;
				}
				camera_tensor = get_tensor_from_camera(c2w, false);
				camera_tensor.to(torch::Device(torch::kCUDA,0));
				camera_tensor.requires_grad_(true);
				camera_tensor_list.push_back(camera_tensor);
			}
		}
		// if BA, assuming its true TODO

	}
	torch::optim::Adam optimizer({decoders_para_list, coarse_grid_para, middle_grid_para, fine_grid_para, color_grid_para, camera_tensor_list}, torch::optim::AdamOptions(0));
	for (int joint_iter=0; joint_iter<num_joint_iters_; joint_iter++)
	{
		if (frustum_feature_selection)
		{
			if (coarse_mapper)
			{
				coarse_val.index_put_({coarse_mask}, coarse_val_grad);
				c.insert("grid_coarse", coarse_val);
			}
			else
			{
				middle_val.index_put_({middle_mask}, middle_val_grad);
				c.insert("grid_middle", middle_val);
				fine_val.index_put_({fine_mask}, fine_val_grad);
				c.insert("grid_fine", fine_val);
				color_val.index_put_({color_mask}, color_val_grad);
				c.insert("grid_color", color_val);
			}

		}
		if (coarse_mapper)
			stage = "coarse";
		else if (joint_iter <= int(num_joint_iters_*middle_iter_ratio))
			stage = "middle";
		else if (joint_iter <= int(num_joint_iters_*fine_iter_ratio))
			stage = "middle";
		else
			stage = "color";

		static_cast<torch::optim::AdamOptions&>(optimizer.param_groups()[0].options()).lr(ns_cfg["mapping"]["stage"][stage]["decoders_lr"].as<float>()*lr_factor);
		static_cast<torch::optim::AdamOptions&>(optimizer.param_groups()[1].options()).lr(ns_cfg["mapping"]["stage"][stage]["coarse_lr"].as<float>()*lr_factor);
		static_cast<torch::optim::AdamOptions&>(optimizer.param_groups()[2].options()).lr(ns_cfg["mapping"]["stage"][stage]["middle_lr"].as<float>()*lr_factor);
		static_cast<torch::optim::AdamOptions&>(optimizer.param_groups()[3].options()).lr(ns_cfg["mapping"]["stage"][stage]["fine_lr"].as<float>()*lr_factor);
		static_cast<torch::optim::AdamOptions&>(optimizer.param_groups()[4].options()).lr(ns_cfg["mapping"]["stage"][stage]["color_lr"].as<float>()*lr_factor);

		if (BA)
			if (stage == "color")
				static_cast<torch::optim::AdamOptions&>(optimizer.param_groups()[5].options()).lr(BA_cam_lr);

		optimizer.zero_grad();
		std::vector<torch::Tensor> batch_rays_d_vec, batch_rays_o_vec, batch_gt_depth_vec, batch_gt_color_vec;
		int camera_tensor_id = 0;
		torch::Tensor gt_depth, gt_color;
		torch::Tensor batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color;

		for (auto& frame : optimize_frame)
		{
			if (frame != -1)
			{
				gt_depth = keyframe_vector[frame].depth.to(torch::Device(torch::kCUDA, 0));
				gt_color = keyframe_vector[frame].color.to(torch::Device(torch::kCUDA, 0));
				if (BA && frame != oldest_frame)
				{
					camera_tensor = camera_tensor_list[camera_tensor_id];
					camera_tensor_id++;
					c2w = get_camera_from_tensor(camera_tensor);
				}
				else
					c2w = keyframe_vector[frame].est_c2w;
			}
			else
			{
				gt_depth = cur_gt_depth.to(torch::Device(torch::kCUDA, 0));
				gt_color = cur_gt_color.to(torch::Device(torch::kCUDA, 0));
				if (BA)
				{
					camera_tensor = camera_tensor_list[camera_tensor_id];
					c2w = get_camera_from_tensor(camera_tensor);
				}
				else
					c2w = cur_c2w;
			}

			get_samples(0, H, 0, W, pixs_per_image, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color);
			batch_rays_o_vec.push_back(batch_rays_o); //TO DO convert to float
			batch_rays_d_vec.push_back(batch_rays_d);
			batch_gt_depth_vec.push_back(batch_gt_depth);
			batch_gt_color_vec.push_back(batch_gt_color);
		}

		batch_rays_d = torch::cat(batch_rays_d_vec);
		batch_rays_o = torch::cat(batch_rays_o_vec);
		batch_gt_depth = torch::cat(batch_gt_depth_vec);
		batch_gt_color = torch::cat(batch_gt_color_vec);

		torch::NoGradGuard noGrad;
		auto det_rays_o = batch_rays_o.unsqueeze(-1);
		auto det_rays_d = batch_rays_d.unsqueeze(-1);

		auto t_ = (bound.unsqueeze(0)-det_rays_o)/det_rays_d;
		auto t = std::get<0>(torch::min(std::get<0>(torch::max(t_, 2)),1));
		t = t.to(torch::Device(torch::kCUDA, 0));
		auto inside_mask = t>=batch_gt_depth;
		batch_rays_d = batch_rays_d.index({inside_mask});
		batch_rays_o = batch_rays_o.index({inside_mask});
		batch_gt_depth = batch_gt_depth.index({inside_mask});
		batch_gt_color = batch_gt_color.index({inside_mask});
		
		torch::Tensor color, depth, uncertainity, weights;
		renderer.render_batch_ray(c, decoders, batch_rays_d.to(torch::Device(torch::kCUDA, 0)), batch_rays_o.to(torch::Device(torch::kCUDA, 0)), "color", batch_gt_depth, color, depth, uncertainity, weights);
		torch::Tensor mask;
		batch_gt_depth = batch_gt_depth.to(torch::Device(torch::kCUDA, 0));
		batch_gt_color = batch_gt_color.to(torch::Device(torch::kCUDA, 0));

		auto depth_mask = batch_gt_depth > 0;
		auto loss = ((torch::abs(batch_gt_depth.index({depth_mask})-depth.index({depth_mask})))).sum();

		if (stage == "color")
		{
			auto color_loss = torch::abs(batch_gt_color - color).sum();
			loss = loss + w_color_loss*color_loss;
		}
		loss.requires_grad_(true);
		loss.backward();
		optimizer.step();
		optimizer.zero_grad();

        if (frustum_feature_selection)
        {
			if (coarse_mapper)
			{
				coarse_val.index_put_({coarse_mask}, coarse_val_grad);
				c.insert("grid_coarse", coarse_val);
			}
			else
			{
				middle_val.index_put_({middle_mask}, middle_val_grad);
				c.insert("grid_middle", middle_val);
				fine_val.index_put_({fine_mask}, fine_val_grad);
				c.insert("grid_fine", fine_val);
				color_val.index_put_({color_mask}, color_val_grad);
				c.insert("grid_color", color_val);
			}
        }
	}

	if (BA)
	{
		int camera_tensor_id = 0;
		for (auto& frame : optimize_frame)
		{
			if (frame != -1)
			{
				if (frame != oldest_frame)
				{
					c2w = get_camera_from_tensor(camera_tensor_list[camera_tensor_id]);
					c2w = torch::cat({c2w, bottom}, 0);
					camera_tensor_id++;
					keyframe_vector[frame].est_c2w = c2w;
				}
			}
			else
			{
					c2w = get_camera_from_tensor(camera_tensor_list[-1]);
					c2w = torch::cat({c2w, bottom}, 0);
					cur_c2w = c2w;
			}
		}
	}

}

void Mapper::run(NICE& decoders, c10::Dict<std::string, torch::Tensor>& c, std::vector<torch::Tensor>& estimate_c2w_vec, torch::Tensor gt_color_t, torch::Tensor gt_depth_t, torch::Tensor gt_c2w_t, int idx, int n_imgs)
{
	bool init = true;
	int prev_idx = -1;
	int outer_joint_iters;
	int lr_factor, num_joint_iters;

	if (!init)
	{
		lr_factor = ns_cfg["mapping"]["lr_factor"].as<int>();
		num_joint_iters = ns_cfg["mapping"]["iters"].as<int>();

		if ((idx == n_imgs-1) && (color_refine) && (!coarse_mapper))
		{
			outer_joint_iters = 5;
			mapping_window_size *= 2;
			middle_iter_ratio = 0.0;
			fine_iter_ratio = 0.0;
			num_joint_iters *= 5;
			fix_color = true;
			frustum_feature_selection = false;
		}
		else
			outer_joint_iters = 1;
	}
	else
	{
		outer_joint_iters = 1;
		lr_factor = ns_cfg["mapping"]["lr_first_factor"].as<int>();
		num_joint_iters = ns_cfg["mapping"]["iters_first"].as<int>();
	}


	torch::Tensor cur_c2w = estimate_c2w_vec[idx];
	num_joint_iters = int(num_joint_iters/outer_joint_iters);
	for (int outer_joint_iter=0; outer_joint_iter<outer_joint_iters; outer_joint_iter++)
	{
		BA = ((keyframe_lvector.size() > 4) && (ns_cfg["mapping"]["BA"].as<bool>()) &&(!coarse_mapper));
		optimize_map(num_joint_iters, c, gt_color_t, gt_depth_t, gt_c2w_t, cur_c2w, decoders);
		// torch::Tensor sz = torch::tensor(c.at("grid_fine").sizes());
		if (BA)
			estimate_c2w_vec[idx] = cur_c2w;

		// // # add new frame to keyframe set
		if (outer_joint_iter == outer_joint_iters-1)
		{
		    if (((idx % keyframe_every == 0) || (idx == n_imgs-2)) && (std::find(keyframe_lvector.begin(), keyframe_lvector.end(), idx) == keyframe_lvector.end()))
		    {
				keyframe_lvector.push_back(idx);
				KeyFrame kf;
				kf.gt_c2w = gt_c2w_t;
				kf.idx = idx;
				kf.color = gt_color_t;
				kf.depth = gt_depth_t;
				kf.est_c2w = cur_c2w;
				keyframe_vector.push_back(kf);
		    }
		}
	}
}


