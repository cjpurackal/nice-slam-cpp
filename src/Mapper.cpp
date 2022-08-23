//cjpurackal
//June 10 '22, 9:00:00

#include "Mapper.h"

Mapper::Mapper(YAML::Node ns_config, YAML::Node cf_config, c10::Dict<std::string, torch::Tensor> c_dict, bool cmapr)
{
	ns_cfg = ns_config;
	cf_cfg = cf_config;
	color_refine = ns_cfg["mapping"]["color_refine"].as<bool>();
	mapping_window_size = ns_cfg["mapping"]["mapping_window_size"].as<int>();
	middle_iter_ratio = ns_cfg["mapping"]["middle_iter_ratio"].as<float>();
	fine_iter_ratio = ns_cfg["mapping"]["fine_iter_ratio"].as<float>();
	fix_color = ns_cfg["mapping"]["fix_color"].as<bool>();
	keyframe_selection_method = ns_cfg["mapping"]["keyframe_selection_method"].as<std::string>();
	frustum_feature_selection = ns_cfg["mapping"]["frustum_feature_selection"].as<bool>();
	BA = false;
	coarse_mapper = cmapr;
	H = cf_config["cam"]["H"].as<int>();
	W = cf_config["cam"]["W"].as<int>();
	fx = cf_config["cam"]["fx"].as<float>();
	fy = cf_config["cam"]["fy"].as<float>();
	cx = cf_config["cam"]["cx"].as<float>();
	cy = cf_config["cam"]["cy"].as<float>();
	c = c_dict;
	mapping_pixels = cf_config["mapping"]["pixels"].as<int>();
	bound = torch::tensor({{-4.5, 3.82},{-1.5, 2.02}, {-3.0, 2.76}});
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
	uv.to(torch::kFloat32);

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
	// # depth test
	// auto mask_tmp = mask_ & (0 <= -1 * z.index({Slice(None), Slice(None), 0})); 
	// auto mask_tmp2 = z.index({Slice(None), Slice(None), 0}) <= depths+0.5;
	// mask_ = mask_.reshape({-1});

	// # add feature grid near cam center
	// auto ray_o_ = c2w_e.topRightCorner(3,1);
	// auto ray_o = torch::from_blob(ray_o_.data(), {3,1}).unsqueeze(0);

	// auto dist_ = points-ray_o;
	// auto dist = torch::sum(dist_*dist_, 1);
	// auto mask2 = dist < 0.5*0.5;
	// mask = mask_ | mask2;

	// points = points.index({mask});
	// mask = mask.reshape({val_shape[2].item<int>(), val_shape[1].item<int>(), val_shape[0].item<int>()});

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

	for (int i=0; i<keyframe_vector.size(); i++)
	{
		c2w = keyframe_vector[i].est_c2w;
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
		uv.to(torch::kFloat32);
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

void Mapper::optimize_map(torch::Tensor cur_gt_color, torch::Tensor cur_gt_depth, torch::Tensor cur_c2w)
{
		// bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
		//     [1, 4])).type(torch.float32).to(device)
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
	if (keyframe_vector.size() > 0)
	{
		optimize_frame.push_back(keyframe_vector.size()-1);
		oldest_frame = *min_element(optimize_frame.begin(), optimize_frame.end());
	}	
	optimize_frame.push_back(-1);

	std::vector<torch::Tensor> coarse_grid_para, middle_grid_para, fine_grid_para, color_grid_para;
	torch::Tensor mask;
	//keyinfo saving to do
	int pixs_per_image = int(mapping_pixels/optimize_frame.size());
	torch::Tensor mask_c2w;
	if (frustum_feature_selection)
		mask_c2w = cur_c2w;
	if (!frustum_feature_selection)
	{
		torch::Tensor coarse_val = c.at("grid_coarse");
		coarse_val.to(torch::Device(torch::kCUDA, 0));
		coarse_val.requires_grad_(true);
		coarse_grid_para.push_back(coarse_val);

		torch::Tensor middle_val = c.at("grid_middle");
		middle_val.to(torch::Device(torch::kCUDA, 0));
		middle_val.requires_grad_(true);
		middle_grid_para.push_back(middle_val);

		torch::Tensor fine_val = c.at("grid_fine");
		fine_val.to(torch::Device(torch::kCUDA, 0));
		fine_val.requires_grad_(true);
		fine_grid_para.push_back(fine_val);

		torch::Tensor color_val = c.at("grid_color");
		color_val.to(torch::Device(torch::kCUDA, 0));
		color_val.requires_grad_(true);
		color_grid_para.push_back(color_val);
	}
	else
	{
		// incomplete
	    cv::Mat depth_mat(cur_gt_depth.sizes()[0], cur_gt_depth.sizes()[1], CV_32FC1);
		std::memcpy(cur_gt_depth.data_ptr(), depth_mat.data, sizeof(float)*cur_gt_depth.numel());

		// torch::Tensor coarse_val = c.at("grid_coarse");
		// get_mask_from_c2w(depth_mat, mask_c2w, torch::tensor(coarse_val.sizes()), "grid_coarse", mask);
		// coarse_val.to(torch::Device(torch::kCUDA, 0));
		// coarse_val.requires_grad_(true);
		// coarse_grid_para.push_back(coarse_val);

		torch::Tensor middle_val = c.at("grid_middle");
		get_mask_from_c2w(depth_mat, mask_c2w, torch::tensor({middle_val.sizes()[2],middle_val.sizes()[3], middle_val.sizes()[4]}), "grid_coarse", mask);

		// middle_val.to(torch::Device(torch::kCUDA, 0));
		// middle_val.requires_grad_(true);
		// middle_grid_para.push_back(middle_val);

		// torch::Tensor fine_val = c.at("grid_fine");
		// fine_val.to(torch::Device(torch::kCUDA, 0));
		// fine_val.requires_grad_(true);
		// fine_grid_para.push_back(fine_val);

		// torch::Tensor color_val = c.at("grid_color");
		// color_val.to(torch::Device(torch::kCUDA, 0));
		// color_val.requires_grad_(true);
		// color_grid_para.push_back(color_val);
	}


}

void Mapper::run(CoFusionReader cfreader) 
{
	bool init = false;
	int idx = 0;
	int prev_idx = -1;
	int outer_joint_iters;
	/*while*/ if(1)
	{
		// while True waiter TODO
		cfreader.getNext();
		auto gt_color = cfreader.rgb;
		auto gt_depth = cfreader.depth; 
		auto gt_color_t = torch::from_blob(gt_color.data, {cfreader.height, cfreader.width, 3}); // 0-1 range 
		auto gt_depth_t = torch::from_blob(gt_depth.data, {cfreader.height, cfreader.width});// 1-2 range observed 
		auto c2w = cfreader.c2w;
		auto gt_c2w_t = torch::from_blob(c2w.data(), {4, 4});

		int lr_factor, num_joint_iters;
		if (!init)
		{
			lr_factor = ns_cfg["mapping"]["lr_factor"].as<int>();
			num_joint_iters = ns_cfg["mapping"]["iters"].as<int>();

			if ((idx == cfreader.n_imgs-1) && (color_refine) && (!coarse_mapper))
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
			num_joint_iters = ns_cfg["mapping"]["num_joint_iters"].as<int>();

		}

		Eigen::MatrixXf I;
		I = Eigen::MatrixXf::Identity(4,4);
		//change this to est_c
		torch::Tensor cur_c2w = torch::from_blob(I.data(), {4,4});
		num_joint_iters = int(num_joint_iters/outer_joint_iters);
		outer_joint_iters = 1;  // delete this
		for (int i=0; i<outer_joint_iters; i++)
		{
			BA = ((keyframe_list.size() > 4) && (ns_cfg["mapping"]["BA"].as<bool>()) &&(!coarse_mapper));
			// _ = self.optimize_map(num_joint_iters, lr_factor, idx, gt_color, gt_depth, gt_c2w, self.keyframe_dict, self.keyframe_list, cur_c2w=cur_c2w)
			optimize_map(gt_color_t, gt_depth_t, gt_c2w_t);
			torch::Tensor sz = torch::tensor(c.at("grid_fine").sizes());
			if (BA)
			{
				// cur_c2w = _
				// self.estimate_c2w_list[idx] = cur_c2w
			}

			// # add new frame to keyframe set
			// if outer_joint_iter == outer_joint_iters-1:
			//     if (idx % self.keyframe_every == 0 or (idx == self.n_img-2)) \
			//             and (idx not in self.keyframe_list):
			//         self.keyframe_list.append(idx)
			//         self.keyframe_dict.append({'gt_c2w': gt_c2w.cpu(), 'idx': idx, 'color': gt_color.cpu(
			//         ), 'depth': gt_depth.cpu(), 'est_c2w': cur_c2w.clone()})

		}


	}
}


