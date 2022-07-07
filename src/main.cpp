//cjpurackal
//June 11 '22, 16:54:00
#include "Tracker.h"

int main(int argc, const char* argv[])
{
	YAML::Node cf_config = YAML::LoadFile("/home/developer/nice-slam-cpp/config/nice_slam.yaml");
	Tracker tracker(cf_config);
	CoFusionReader cfreader("/home/developer/nice-slam-cpp/Datasets/CoFusion/room4/");


	int dim, c_dim, hidden_size;
	float coarse_grid_len, middle_grid_len, fine_grid_len, color_grid_len;
	bool coarse;
	std::string pose_emb;

	dim = cf_config["data"]["dim"].as<int>();
	coarse_grid_len = cf_config["grid_len"]["coarse"].as<float>();
	middle_grid_len = cf_config["grid_len"]["middle"].as<float>();
	fine_grid_len = cf_config["grid_len"]["fine"].as<float>();
	color_grid_len = cf_config["grid_len"]["color"].as<float>();
	c_dim = cf_config["model"]["c_dim"].as<float>();
	pose_emb = cf_config["model"]["pos_embedding_method"].as<std::string>();
	hidden_size = 32;

	NICE decoders(dim, c_dim, hidden_size, coarse_grid_len, middle_grid_len, fine_grid_len, color_grid_len, coarse, pose_emb);

	tracker.run(cfreader, decoders);


	return 0;
}