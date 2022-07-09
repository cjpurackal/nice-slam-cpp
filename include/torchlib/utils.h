#include <torch/torch.h>
#include <torch/script.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace torch::indexing;
namespace F = torch::nn::functional;
typedef std::pair<int, float> pair;

inline void raySampler(int H0, int H1, int W0, int W1, int n, int fx, int fy, int cx, int cy, torch::Tensor depth, torch::Tensor color, torch::Tensor c2w, torch::Tensor& rays_d, torch::Tensor& rays_o, torch::Tensor& gt_color, torch::Tensor& gt_depth)
{

	depth = depth.index({Slice(H0,H1), Slice(W0,W1)});
	color = color.index({Slice(H0,H1), Slice(W0,W1)});


	std::vector<torch::Tensor> tl;
	tl.push_back(torch::linspace(W0,W1-1,W1-W0,torch::TensorOptions().dtype(torch::kF32)/*.device(torch::kCUDA, 0)*/));
	tl.push_back(torch::linspace(H0,H1-1,H1-H0,torch::TensorOptions().dtype(torch::kF32)/*.device(torch::kCUDA, 0)*/));
	std::vector<torch::Tensor> t = torch::meshgrid(tl);
	torch::Tensor i = t[0];
	torch::Tensor j = t[1];

	i = i.t();
	j = j.t();

	i = i.reshape(-1);
	j = j.reshape(-1);

	torch::Tensor ind = torch::randint(i.sizes()[0], {n});
	ind = ind.clamp(0, i.sizes()[0]);
	ind = ind.to(torch::kLong);
	i = i.index({ind});
	j = j.index({ind});

	depth = depth.reshape(-1);
	color = color.reshape({-1, 3});

	gt_color = color.index({ind});
	gt_depth = depth.index({ind});

	auto i_t = (i-cx)/fx;
	auto j_t = (i-cy)/fy;

	auto dirs = torch::stack({i_t,j_t,-torch::ones_like(i)}, -1);
	dirs = dirs.reshape({-1, 1, 3});

	rays_d = torch::sum(dirs * c2w.index({Slice(None,3), Slice(None, 3)}), -1);
	rays_o = c2w.index({Slice(None, 3), -1}).expand(rays_d.sizes());

}


inline void keyframe_selection_overlap(int H0, int H1, int W0, int W1, int n, int fx, int fy, int cx, int cy, torch::Tensor rgb_, torch::Tensor depth_, torch::Tensor c2w_)
{

	int n_samples =16;
	int W =W1;
	int H = H1;
	int k_overlap = 0;

	torch::Tensor rays_o, rays_d, gt_color, gt_depth;
	raySampler(H0, H1, W0, W1, n, fx, fy, cx, cy, depth_, rgb_, c2w_, rays_d, rays_o, gt_color, gt_depth);

	gt_depth = gt_depth.reshape({-1, 1});
	gt_depth = gt_depth.repeat({1,n_samples});
	auto t_vals = torch::linspace(0, 1, n_samples);
	auto near = gt_depth * 0.8;
	auto far = gt_depth + 0.5;
	auto z_vals = near * (1 - t_vals) + far * (t_vals);
	auto pts = rays_o.index({"...", None, Slice(None)}) + rays_d.index({"...", None, Slice(None)}) * z_vals.index({"...", Slice(None), None});
	auto vertices = pts.reshape({-1, 3});

	std::vector<Eigen::Matrix4f> keyframe_vec;
	keyframe_vec.push_back(Eigen::Matrix4f::Identity(4,4));

	Eigen::Matrix3f k;
	k << fx, 0, cx,
		0, fy, cy,
		0, 0, 1;
	auto K = torch::from_blob(k.data(), {3, 3});
	std::map<int, float> list_keyframe;

	float best_p = 0;
	for (int i = 0; i < keyframe_vec.size(); ++i)
	{
		Eigen::Matrix4f c2w = keyframe_vec[i];
		Eigen::Matrix4f w2c_ = c2w.inverse();
		auto w2c = torch::from_blob(w2c_.data(), {4,4});
		auto ones = torch::ones_like(vertices.index({Slice(None), 0})).reshape({-1, 1});
		auto homo_vertices = torch::cat({vertices, ones}, 1).reshape({-1, 4, 1});
		auto cam_cord_homo = torch::matmul(w2c, homo_vertices);
		auto cam_cord = cam_cord_homo.index({Slice(None), Slice(None, 3)});

		cam_cord.index({Slice(None), 0}) = cam_cord.index({Slice(None), 0}) * -1;
		auto uv = torch::matmul(K, cam_cord);
		auto z = uv.index({Slice(None), Slice(-1, None)})+1e-5;
		uv = uv.index({Slice(None), Slice(None, 2)})/z;
		//uv to float conver here TODO
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
    std::vector<int> selected_kf;
    for (auto ele : sorted_kf)
    	selected_kf.push_back(ele.first);

    if ((selected_kf.size()-k_overlap) > 0)
    	selected_kf.assign(selected_kf.begin(), selected_kf.begin()+k_overlap);
}


inline void normalize_3d_coordinate(torch::Tensor& p, torch::Tensor bound)
{
    p = p.reshape({-1, 3});
    p.index({Slice(None), 0}) = ((p.index({Slice(None, 0)}) - bound.index({0,0})) / (bound.index({0,1}) - bound.index({0, 0})))*2-1; 
    p.index({Slice(None), 1}) = ((p.index({Slice(None, 1)}) - bound.index({1,0})) / (bound.index({1,1}) - bound.index({1, 0})))*2-1; 
    p.index({Slice(None), 2}) = ((p.index({Slice(None, 2)}) - bound.index({2,0})) / (bound.index({2,1}) - bound.index({2, 0})))*2-1; 

}

inline void get_samples(int H0, int H1, int W0, int W1, int n, int H, int W, int fx, int fy, int cx, int cy, torch::Tensor c2w, torch::Tensor depth, torch::Tensor color, torch::Tensor& rays_d, torch::Tensor& rays_o, torch::Tensor& sample_depth, torch::Tensor& sample_color)
{

	raySampler(H0, H1, W0, W1, n, fx, fy, cx, cy, depth, color, c2w, rays_d, rays_o, sample_color, sample_depth);

}

inline void raw2outputs_nerf_color(torch::Tensor raw, torch::Tensor z_vals, bool occupancy, torch::Tensor rays_d, torch::Tensor& rgb_map, torch::Tensor& depth_map, torch::Tensor& depth_var, torch::Tensor& weights)
{
	auto dists = z_vals.index({"...", Slice(1, None)}) - z_vals.index({"...", Slice(-1, None)});
	dists = torch::cat({dists, torch::tensor({1e10}).expand(dists.index({"...", Slice(None, 1)}).sizes())}, -1);
	dists = dists * torch::norm(rays_d.index({"...", None, Slice(None)}), -1);
	auto rgb = raw.index({"...", Slice(None, -1)});
	//assuming occupancy is false
	auto alpha = 1 - torch::exp(-F::relu(raw.index({"...", -1}))*dists);

	weights = alpha * torch::cumprod(
										torch::cat({
											torch::ones({alpha.sizes()[0], 1})
											,1-alpha + 1e-10
										},-1)
									, -1).index({Slice(None), Slice(None, -1)}); 

	rgb_map = torch::sum(weights.index({"...", Slice(None)}) * rgb, -2);
	depth_map = torch::sum(weights * z_vals, -1);
	auto tmp = (z_vals - depth_map.unsqueeze(-1));
	depth_var = torch::sum(weights*tmp*tmp, 1);
}

inline torch::Tensor quad2rotation(torch::Tensor quad)
{
	int bs = quad.sizes()[0];
	auto qr = quad.index({Slice(None), 0});
	auto qi = quad.index({Slice(None), 1});
	auto qj = quad.index({Slice(None), 2});
	auto qk = quad.index({Slice(None), 3});
	auto two_s = 2/(quad*quad).sum(-1);
	auto rot_mat = torch::zeros({bs, 3, 3});
	rot_mat.index({Slice(None), 0, 0}) = 1 - two_s * (qj.pow(2) + qk.pow(2));
	rot_mat.index({Slice(None), 0, 1}) = two_s * (qi * qj - qk * qr);
	rot_mat.index({Slice(None), 0, 2}) = two_s * (qi * qk + qj * qr);
	rot_mat.index({Slice(None), 1, 0}) = two_s * (qi * qj + qk * qr);
	rot_mat.index({Slice(None), 1, 1}) = 1 - two_s * (qi.pow(2) + qk.pow(2));
	rot_mat.index({Slice(None), 1, 2}) = two_s * (qj * qk - qi * qr);
	rot_mat.index({Slice(None), 2, 0}) = two_s * (qi * qk - qj * qr);
	rot_mat.index({Slice(None), 2, 1}) = two_s * (qj * qk + qi * qr);
	rot_mat.index({Slice(None), 2, 2}) = 1 - two_s * (qi.pow(2) + qj.pow(2));

	return rot_mat;

}


inline torch::Tensor get_camera_from_tensor(torch::Tensor inputs)
{
	int n = inputs.sizes().size();
	if (n == 1)
		inputs = inputs.unsqueeze(0);
	auto quad = inputs.index({Slice(None), Slice(None, 4)});
	auto T = inputs.index({Slice(None), Slice(4, None)});	
	auto R = quad2rotation(quad);
	auto RT = torch::cat({R, T.index({Slice(None), Slice(None), None})}, 2);
	if (n == 1)
		RT = RT[0]; //not sure if this works
	return RT;
}

inline torch::Tensor get_tensor_from_camera(torch::Tensor RT, bool Tquad)
{

	typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_rm; // same as MatrixXf, but with row-major memory layout
    auto R = RT.index({Slice(None,3), Slice(None,3)});
    auto T = RT.index({Slice(None,3), 3});
    R.data<float>();
    Eigen::Map<MatrixXf_rm> rot(R.data_ptr<float>(), R.size(0), R.size(1));
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot_ = Eigen::Matrix3f::Identity(3,3);
    Eigen::Quaternionf quad(rot_);
    torch::Tensor quad_t = torch::tensor({quad.x(), quad.y(), quad.z(), quad.w()});
    torch::Tensor tensor;
    
    if (Tquad)
    	tensor = torch::cat({T, quad_t}, 0);
    else
    	tensor = torch::cat({quad_t, T}, 0);

    return tensor;
}
