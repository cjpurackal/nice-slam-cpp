#include <torch/torch.h>

struct DenseLayer: torch::nn::Linear
{
	DenseLayer()
	{

	}
};

// def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", *args, **kwargs) -> None:
//     self.activation = activation
//     super().__init__(in_dim, out_dim, *args, **kwargs)

// def reset_parameters(self) -> None:
//     torch.nn.init.xavier_uniform_(
//         self.weight, gain=torch.nn.init.calculate_gain(self.activation))
//     if self.bias is not None:
//         torch.nn.init.zeros_(self.bias)