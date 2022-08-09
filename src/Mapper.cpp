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
	frustum_feature_selection = ns_cfg["mapping"]["frustum_feature_selection"].as<bool>();
	BA = false;
	coarse_mapper = cmapr;
}
Mapper::~Mapper()
{

}
void Mapper::run(CoFusionReader cfreader) 
{
	bool init = false;
	int idx = 0;
	int prev_idx = -1;
	int outer_joint_iters;
	while(1)
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
		for (int i=0; i<outer_joint_iters; i++)
		{
			BA = ((keyframe_list.size() > 4) && (ns_cfg["mapping"]["BA"].as<bool>()) &&(!coarse_mapper));
			// _ = self.optimize_map(num_joint_iters, lr_factor, idx, gt_color, gt_depth, gt_c2w, self.keyframe_dict, self.keyframe_list, cur_c2w=cur_c2w)
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


