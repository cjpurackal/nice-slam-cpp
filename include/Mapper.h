//cjpurackal
//June 11 '22, 16:43:00

#include <iostream>
#include <memory>
#include "inputs/CoFusionReader.h"
#include <yaml-cpp/yaml.h>
#include "Renderer.h"

class Mapper
{
	public:
		Mapper(YAML::Node ns_config, YAML::Node cf_config, c10::Dict<std::string, torch::Tensor> c_dict, bool coarse_mapper);
		virtual ~Mapper();
		void run(CoFusionReader cfreader);
		torch::Tensor optimize_map();
		torch::Tensor get_mask_from_c2w();
	private:
		YAML::Node ns_cfg, cf_cfg;
		bool color_refine, coarse_mapper, fix_color, frustum_feature_selection, BA;
		int mapping_window_size;
		float middle_iter_ratio, fine_iter_ratio;
		std::vector<int> keyframe_list;

};