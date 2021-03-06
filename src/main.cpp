//cjpurackal
//June 11 '22, 16:54:00
#include "Tracker.h"

int main(int argc, const char* argv[])
{
	YAML::Node ns_config = YAML::LoadFile("/home/developer/nice-slam-cpp/config/nice_slam.yaml");
	YAML::Node cf_config = YAML::LoadFile("/home/developer/nice-slam-cpp/config/cofusion.yaml");

	torch::DeviceType device_type;
	device_type = torch::kCUDA;
	torch::Device device0(device_type, 0);

	CoFusionReader cfreader("/home/developer/nice-slam-cpp/Datasets/CoFusion/room4/");


	int dim, c_dim, hidden_size, coarse_bound_enlarge;
	float coarse_grid_len, middle_grid_len, fine_grid_len, color_grid_len;
	bool coarse;
	std::string pose_emb;

	dim = ns_config["data"]["dim"].as<int>();
	coarse_grid_len = ns_config["grid_len"]["coarse"].as<float>();
	middle_grid_len = ns_config["grid_len"]["middle"].as<float>();
	fine_grid_len = ns_config["grid_len"]["fine"].as<float>();
	color_grid_len = ns_config["grid_len"]["color"].as<float>();
	c_dim = ns_config["model"]["c_dim"].as<float>();
	pose_emb = ns_config["model"]["pos_embedding_method"].as<std::string>();
	hidden_size = 32;
	coarse_bound_enlarge = ns_config["model"]["coarse_bound_enlarge"].as<int>();
	coarse_bound_enlarge = ns_config["model"]["coarse_bound_enlarge"].as<int>();
	coarse_bound_enlarge = ns_config["model"]["coarse_bound_enlarge"].as<int>();
	auto bound = torch::tensor({{-4.5, 3.82},{-1.5, 2.02}, {-3.0, 2.76}});
	auto xyz_len = bound.index({Slice(None), 1}) - bound.index({Slice(None), 0});
	c10::Dict<std::string, torch::Tensor> c_dict;
	std::vector<int64_t> coarse_val_shape, middle_val_shape, fine_val_shape, color_val_shape; 

	auto coarse_val_shape_ = xyz_len*coarse_bound_enlarge/coarse_grid_len;
	coarse_val_shape.push_back(1);
	coarse_val_shape.push_back(c_dim);
	coarse_val_shape.push_back(coarse_val_shape_[2].item<int>());
	coarse_val_shape.push_back(coarse_val_shape_[1].item<int>());
	coarse_val_shape.push_back(coarse_val_shape_[0].item<int>());
	auto coarse_val = torch::zeros(coarse_val_shape).normal_(0, 0.01);
	coarse_val = coarse_val.to(device0);
	c_dict.insert(std::string("grid_coarse"), coarse_val);

	auto middle_val_shape_ = xyz_len/middle_grid_len;
	middle_val_shape.push_back(1);
	middle_val_shape.push_back(c_dim);
	middle_val_shape.push_back(middle_val_shape_[2].item<int>());
	middle_val_shape.push_back(middle_val_shape_[1].item<int>());
	middle_val_shape.push_back(middle_val_shape_[0].item<int>());
	auto middle_val = torch::zeros(middle_val_shape).normal_(0, 0.01);
	middle_val = middle_val.to(device0);
	c_dict.insert(std::string("grid_middle"), middle_val);


	auto fine_val_shape_ = xyz_len/fine_grid_len;
	fine_val_shape.push_back(1);
	fine_val_shape.push_back(c_dim);
	fine_val_shape.push_back(fine_val_shape_[2].item<int>());
	fine_val_shape.push_back(fine_val_shape_[1].item<int>());
	fine_val_shape.push_back(fine_val_shape_[0].item<int>());
	auto fine_val = torch::zeros(fine_val_shape).normal_(0, 0.0001);
	fine_val = fine_val.to(device0);
	c_dict.insert(std::string("grid_fine"), fine_val);


	auto color_val_shape_ = xyz_len/color_grid_len;
	color_val_shape.push_back(1);
	color_val_shape.push_back(c_dim);
	color_val_shape.push_back(color_val_shape_[2].item<int>());
	color_val_shape.push_back(color_val_shape_[1].item<int>());
	color_val_shape.push_back(color_val_shape_[0].item<int>());
	auto color_val = torch::zeros(color_val_shape).normal_(0, 0.01);
	color_val = color_val.to(device0);
	c_dict.insert(std::string("grid_color"), color_val);


	NICE decoders(dim, c_dim, hidden_size, coarse_grid_len, middle_grid_len, fine_grid_len, color_grid_len, coarse, pose_emb);
	Tracker tracker(ns_config, cf_config, c_dict);


	// auto p = torch::randn({1,10,3});
	// p = p.to(device0);

	// auto out = decoders.forward(p, c_dict, "coarse");
	// std::cout<<out<<std::endl;
	// out = decoders.forward(p, c_dict, "middle");
	// std::cout<<out<<std::endl;
	// out = decoders.forward(p, c_dict, "fine");
	// std::cout<<out<<std::endl;
	// out = decoders.forward(p, c_dict, "color");
	// std::cout<<out<<std::endl;
	tracker.run(cfreader, decoders);

	return 0;
}



void test_cgird(c10::Dict<std::string, torch::Tensor> cd)
{
	

}