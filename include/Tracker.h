//cjpurackal
//June 11 '22, 16:54:00
#include <iostream>
#include "inputs/CoFusionReader.h"
#include "Renderer.h"
#include <yaml-cpp/yaml.h>

class Tracker
{
	public:
		Tracker(YAML::Node ns_config, YAML::Node cf_config);
		virtual ~Tracker();
		void run(CoFusionReader cfreaderm , NICE decoders);
		torch::Tensor optimize_cam_in_batch(torch::Tensor cam_tensor, torch::Tensor gt_color, torch::Tensor gt_depth, int batch_size , torch::optim::Adam& optimizer, NICE decoders);
		void update_para_from_mapping();
	private:
		int H, W;
		float fx, fy, cx, cy;
		int idx, ignore_edge_w, ignore_edge_h;
		torch::Tensor bound;
		Renderer renderer;
		std::map<std::string, torch::Tensor> c;
		bool handle_dynamic, use_color_in_tracking;
		float w_color_loss;
		int mapping_idx, prev_mapping_idx;
		std::map<std::string, torch::Tensor> shared_c;
		float lr;
		std::vector<torch::Tensor> estimate_c2w; 
		float current_min_loss;
		int num_cam_iters, tracking_pixels;
};