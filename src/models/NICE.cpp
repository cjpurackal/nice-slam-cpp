#include "models/NICE.h"

NICE::NICE(int dim, int c_dim, int hidden_size, float coarse_grid_len, float middle_grid_len, float fine_grid_len, float color_grid_len, bool coarse, std::string pose_emb):
coarse_decoder("coarse", dim, c_dim, hidden_size, 5, false, std::vector<int>({2}), coarse_grid_len),
middle_decoder("middle", dim, c_dim, hidden_size, 5, false, std::vector<int>({2}), middle_grid_len, pose_emb, false),
fine_decoder("fine", dim, c_dim*2, hidden_size, 5, false, std::vector<int>({2}), fine_grid_len, pose_emb, true),
color_decoder("color", dim, c_dim, hidden_size, 5, true, std::vector<int>({2}), color_grid_len, pose_emb, false),
cmodule(torch::jit::load("/home/developer/nice-slam-cpp/traced/traced_coarse_model.pt")),
mmodule(torch::jit::load("/home/developer/nice-slam-cpp/traced/traced_middle_model.pt")),
fmodule(torch::jit::load("/home/developer/nice-slam-cpp/traced/traced_fine_model.pt")),
colormodule(torch::jit::load("/home/developer/nice-slam-cpp/traced/traced_color_model.pt"))
{

}

torch::Tensor NICE::forward(torch::Tensor p, c10::Dict<std::string, torch::Tensor> c_grid, std::string stage)
{
	if (stage == "coarse")
	{
		auto occ = cmodule.forward({p, c_grid}).toTensor();
		occ = occ.squeeze(0);
		auto raw = torch::zeros({occ.sizes()[0], 4});
		raw.index_put_({"...", -1}, occ);
		return raw;
	}
	else if (stage == "middle")
	{
		auto middle_occ = mmodule.forward({p, c_grid}).toTensor();
		middle_occ = middle_occ.squeeze(0);
		auto raw = torch::zeros({middle_occ.sizes()[0], 4});
		raw.index_put_({"...", -1}, middle_occ);
		return raw;
	}
	else if (stage == "fine")
	{
		auto fine_occ = fmodule.forward({p, c_grid}).toTensor();
		auto raw = torch::zeros({fine_occ.sizes()[0], 4});
		auto middle_occ = mmodule.forward({p, c_grid}).toTensor();
		middle_occ = middle_occ.squeeze(0);
		raw.index_put_({"...", -1}, fine_occ+middle_occ);
		return raw;
	}
	else if (stage == "color")
	{
		auto fine_occ = fmodule.forward({p, c_grid}).toTensor();
		auto raw = colormodule.forward({p, c_grid}).toTensor();
		auto middle_occ = mmodule.forward({p, c_grid}).toTensor();
		middle_occ = middle_occ.squeeze(0);
		raw.index_put_({"...", -1}, fine_occ+middle_occ);
		return raw;
	}
}