//cjpurackal
//June 11 '22, 16:43:00

#include <iostream>
#include <memory>
#include "inputs/CoFusionReader.h"
#include <yaml-cpp/yaml.h>
#include "Renderer.h"
#include <algorithm>

struct KeyFrame
{
	torch::Tensor cur_c2w, est_c2w, gt_c2w, gt_color, gt_depth, color, depth;
	int idx;
};

class Mapper
{
	public:
		Mapper(YAML::Node ns_config, YAML::Node cf_config, c10::Dict<std::string, torch::Tensor> c_dict, bool coarse_mapper);
		virtual ~Mapper();
		void run(CoFusionReader cfreader, NICE& decoders, std::vector<torch::Tensor>& estimate_c2w_vec);
		void optimize_map(torch::Tensor cur_gt_color, torch::Tensor cur_gt_depth, torch::Tensor gt_cur_c2w,  torch::Tensor& cur_c2w, NICE& decoders);
		void keyframe_selection_overlap(torch::Tensor gt_color_, torch::Tensor gt_depth_, torch::Tensor c2w, std::vector<KeyFrame> keyframe_vector_, int k_overlap, std::vector<int>& selected_kf);
		void get_mask_from_c2w(cv::Mat depth_mat, torch::Tensor c2w, torch::Tensor val_shape, std::string key, torch::Tensor& mask);
	private:
		Renderer renderer;
		YAML::Node ns_cfg, cf_cfg;
		bool color_refine, coarse_mapper, fix_color, frustum_feature_selection, BA;
		c10::Dict<std::string, torch::Tensor> c;
		int mapping_window_size, mapping_pixels;
		float middle_iter_ratio, fine_iter_ratio;
		std::vector<int> keyframe_list;
		int H, W;
		float fx, fy, cx, cy;
		std::vector<KeyFrame> keyframe_vector;
		std::string keyframe_selection_method;
		torch::Tensor bound;
		bool fix_fine;
		int num_joint_iters, keyframe_every;
		std::string stage;
		float lr_factor, BA_cam_lr;
		float w_color_loss;


};