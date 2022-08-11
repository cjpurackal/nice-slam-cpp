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
}
Mapper::~Mapper()
{

}

void Mapper::keyframe_selection_overlap(torch::Tensor gt_color_, torch::Tensor gt_depth_, torch::Tensor c2w, std::vector<KeyFrame> keyframe_vector_, int k)
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
	vertices = pts.reshape({-1, 3});


	for (int i=0; i<keyframe_vector.size(); i++)
	{
		c2w = keyframe_vector[0].est_c2w;

	}



	// for keyframeid, keyframe in enumerate(keyframe_dict):
	//     c2w = keyframe['est_c2w'].cpu().numpy()
	//     w2c = np.linalg.inv(c2w)
	//     ones = np.ones_like(vertices[:, 0]).reshape(-1, 1)
	//     homo_vertices = np.concatenate(
	//         [vertices, ones], axis=1).reshape(-1, 4, 1)  # (N, 4)
	//     cam_cord_homo = w2c@homo_vertices  # (N, 4, 1)=(4,4)*(N, 4, 1)
	//     cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)
	//     K = np.array([[fx, .0, cx], [.0, fy, cy],
	//                  [.0, .0, 1.0]]).reshape(3, 3)
	//     cam_cord[:, 0] *= -1
	//     uv = K@cam_cord
	//     z = uv[:, -1:]+1e-5
	//     uv = uv[:, :2]/z
	//     uv = uv.astype(np.float32)
	//     edge = 20
	//     mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
	//         (uv[:, 1] < H-edge)*(uv[:, 1] > edge)
	//     mask = mask & (z[:, :, 0] < 0)
	//     mask = mask.reshape(-1)
	//     percent_inside = mask.sum()/uv.shape[0]
	//     list_keyframe.append(
	//         {'id': keyframeid, 'percent_inside': percent_inside})

	// list_keyframe = sorted(
	//     list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
	// selected_keyframe_list = [dic['id']
	//                           for dic in list_keyframe if dic['percent_inside'] > 0.00]
	// selected_keyframe_list = list(np.random.permutation(
	//     np.array(selected_keyframe_list))[:k])
	// return selected_keyframe_list
}

void Mapper::optimize_map(torch::Tensor cur_gt_color, torch::Tensor cur_gt_depth, torch::Tensor cur_c2w)
{
        // bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
        //     [1, 4])).type(torch.float32).to(device)
	std::vector<KeyFrame> optimize_frame;
    torch::Tensor bottom = torch::tensor({0,0,0,1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    int num;
    if (keyframe_vector.size() != 0)
    {
    	//assuming selection metho to be overlap
        num = mapping_window_size-2;
        keyframe_selection_overlap(cur_gt_color, cur_gt_depth, cur_c2w, keyframe_vector/*[:-1]*/, num);
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


