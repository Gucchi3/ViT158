/*
 * qkv_kernel.cpp
 * PyTorch C++ バインディング — 3値量子化 QKV GEMM カーネル
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

// CUDA カーネルの前方宣言
void ternary_qkv_gemm_cuda(
    const int8_t* A, const int8_t* W, float* C,
    int M, int N, int K, cudaStream_t stream);

/*
 * ternary_qkv_gemm(activation, weight) -> Tensor
 *   activation : (M, K)  torch.int8   — 量子化済み活性化
 *   weight     : (N, K)  torch.int8   — 3値量子化重み
 *   戻り値     : (M, N)  torch.float  — 行列積結果
 */
torch::Tensor ternary_qkv_gemm(torch::Tensor activation, torch::Tensor weight) {
    TORCH_CHECK(activation.is_cuda(), "activation must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(),     "weight must be a CUDA tensor");
    TORCH_CHECK(activation.dtype() == torch::kInt8, "activation must be int8");
    TORCH_CHECK(weight.dtype()     == torch::kInt8, "weight must be int8");

    activation = activation.contiguous();
    weight     = weight.contiguous();

    int M = activation.size(0);
    int K = activation.size(1);
    int N = weight.size(0);
    TORCH_CHECK(weight.size(1) == K, "dimension mismatch");

    auto output = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(activation.device()));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    ternary_qkv_gemm_cuda(
        activation.data_ptr<int8_t>(),
        weight.data_ptr<int8_t>(),
        output.data_ptr<float>(),
        M, N, K, stream);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ternary_qkv_gemm", &ternary_qkv_gemm,
          "Ternary QKV GEMM — int8 activation x ternary weight (CUDA)");
}
