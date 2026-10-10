#include "compat/cuda_api.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

extern "C" {
CUresult cuInit(unsigned int flags);
CUresult cuModuleLoadData(CUmodule *module, const void *image);
CUresult cuModuleGetFunction(CUfunction *function, CUmodule module,
                             const char *name);
CUresult cuMemAlloc(CUdeviceptr *ptr, std::size_t size);
CUresult cuMemFree(CUdeviceptr ptr);
CUresult cuMemcpyHtoD(CUdeviceptr dst, const void *src, std::size_t count);
CUresult cuMemcpyDtoH(void *dst, CUdeviceptr src, std::size_t count);
CUresult cuLaunchKernel(CUfunction function, unsigned int grid_x,
                        unsigned int grid_y, unsigned int grid_z,
                        unsigned int block_x, unsigned int block_y,
                        unsigned int block_z, unsigned int shared_mem,
                        CUstream stream, void **params, void **extra);
}

namespace {

std::vector<float> read_floats(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  std::vector<float> values;
  float value = 0.0F;
  while (input.read(reinterpret_cast<char *>(&value), sizeof(value))) {
    values.push_back(value);
  }
  return values;
}

std::vector<int> read_ints(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  std::vector<int> values;
  int value = 0;
  while (input.read(reinterpret_cast<char *>(&value), sizeof(value))) {
    values.push_back(value);
  }
  return values;
}

void write_floats(const std::string &path, const std::vector<float> &values) {
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
}

CUfunction load_function(const std::string &ptx_path, const std::string &name) {
  std::ifstream ptx_input(ptx_path);
  if (!ptx_input) {
    return nullptr;
  }
  std::ostringstream buffer;
  buffer << ptx_input.rdbuf();
  const std::string ptx = buffer.str();

  if (cuInit(0) != CUDA_SUCCESS) {
    return nullptr;
  }
  CUmodule module = nullptr;
  if (cuModuleLoadData(&module, ptx.c_str()) != CUDA_SUCCESS) {
    return nullptr;
  }
  CUfunction function = nullptr;
  if (cuModuleGetFunction(&function, module, name.c_str()) != CUDA_SUCCESS) {
    return nullptr;
  }
  return function;
}

CUdeviceptr alloc_copy(const void *data, std::size_t bytes) {
  CUdeviceptr ptr = 0;
  if (cuMemAlloc(&ptr, bytes) != CUDA_SUCCESS) {
    return 0;
  }
  if (bytes != 0 && cuMemcpyHtoD(ptr, data, bytes) != CUDA_SUCCESS) {
    cuMemFree(ptr);
    return 0;
  }
  return ptr;
}

bool launch_1d(CUfunction function, const std::vector<void *> &params, int n,
               int block = 256) {
  const unsigned int blocks =
      (static_cast<unsigned int>(n) + static_cast<unsigned int>(block) - 1) /
      static_cast<unsigned int>(block);
  return cuLaunchKernel(function, blocks, 1, 1,
                        static_cast<unsigned int>(block), 1, 1, 0, nullptr,
                        const_cast<void **>(params.data()),
                        nullptr) == CUDA_SUCCESS;
}

bool launch_2d(CUfunction function, const std::vector<void *> &params, int gx,
               int gy, int bx, int by) {
  return cuLaunchKernel(
             function, static_cast<unsigned int>(gx),
             static_cast<unsigned int>(gy), 1, static_cast<unsigned int>(bx),
             static_cast<unsigned int>(by), 1, 0, nullptr,
             const_cast<void **>(params.data()), nullptr) == CUDA_SUCCESS;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    return 1;
  }

  const std::string mode = argv[1];
  const std::string out_path = argv[argc - 1];

  if (mode == "bias_add") {
    if (argc != 7) {
      return 2;
    }
    auto input = read_floats(argv[2]);
    auto bias = read_floats(argv[3]);
    const int n = std::stoi(argv[4]);
    const int channels = std::stoi(argv[5]);
    const std::size_t bytes = input.size() * sizeof(float);
    CUdeviceptr d_in = alloc_copy(input.data(), bytes);
    CUdeviceptr d_bias = alloc_copy(bias.data(), bias.size() * sizeof(float));
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, bytes);
    int count = n;
    int c = channels;
    int per = n / channels;
    (void)c;
    (void)per;
    std::vector<void *> params = {&d_in, &d_bias, &d_out, &count};
    CUfunction f = load_function(BIAS_ADD_PTX_PATH, "bias_add");
    if (!f || !launch_1d(f, params, n)) {
      return 3;
    }
    std::vector<float> output(input.size());
    cuMemcpyDtoH(output.data(), d_out, bytes);
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "batch_norm") {
    if (argc != 9) {
      return 4;
    }
    auto input = read_floats(argv[2]);
    auto scale = read_floats(argv[3]);
    auto bias = read_floats(argv[4]);
    const int channels = std::stoi(argv[5]);
    const int size = std::stoi(argv[6]);
    const float eps = std::stof(argv[7]);
    const int n = channels * size;
    const std::size_t bytes = input.size() * sizeof(float);
    CUdeviceptr d_in = alloc_copy(input.data(), bytes);
    CUdeviceptr d_scale =
        alloc_copy(scale.data(), scale.size() * sizeof(float));
    CUdeviceptr d_bias = alloc_copy(bias.data(), bias.size() * sizeof(float));
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, bytes);
    int c = channels;
    int s = size;
    float e = eps;
    std::vector<void *> params = {&d_in, &d_scale, &d_bias, &d_out, &c, &s, &e};
    CUfunction f = load_function(BATCH_NORM_PTX_PATH, "batch_norm");
    if (!f || !launch_1d(f, params, n)) {
      return 5;
    }
    std::vector<float> output(input.size());
    cuMemcpyDtoH(output.data(), d_out, bytes);
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "max_pool2d" || mode == "avg_pool2d") {
    if (argc != 9) {
      return 6;
    }
    auto input = read_floats(argv[2]);
    const int h = std::stoi(argv[3]);
    const int w = std::stoi(argv[4]);
    const int out_h = std::stoi(argv[5]);
    const int out_w = std::stoi(argv[6]);
    const int kernel = std::stoi(argv[7]);
    const std::size_t out_count =
        static_cast<std::size_t>(out_h) * static_cast<std::size_t>(out_w);
    CUdeviceptr d_in = alloc_copy(input.data(), input.size() * sizeof(float));
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, out_count * sizeof(float));
    int hh = h;
    int ww = w;
    int oh = out_h;
    int ow = out_w;
    int k = kernel;
    std::vector<void *> params = {&d_in, &d_out, &hh, &ww, &oh, &ow, &k};
    const std::string name = mode == "max_pool2d" ? "max_pool2d" : "avg_pool2d";
    const std::string ptx =
        mode == "max_pool2d" ? MAX_POOL_PTX_PATH : AVG_POOL_PTX_PATH;
    CUfunction f = load_function(ptx, name);
    if (!f ||
        !launch_2d(f, params, (out_w + 15) / 16, (out_h + 15) / 16, 16, 16)) {
      return 7;
    }
    std::vector<float> output(out_count);
    cuMemcpyDtoH(output.data(), d_out, out_count * sizeof(float));
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "layer_norm") {
    if (argc != 7) {
      return 8;
    }
    auto input = read_floats(argv[2]);
    const int rows = std::stoi(argv[3]);
    const int hidden = std::stoi(argv[4]);
    const float eps = std::stof(argv[5]);
    const std::size_t bytes = input.size() * sizeof(float);
    CUdeviceptr d_in = alloc_copy(input.data(), bytes);
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, bytes);
    int r = rows;
    int h = hidden;
    float e = eps;
    std::vector<void *> params = {&d_in, &d_out, &r, &h, &e};
    CUfunction f = load_function(LAYER_NORM_PTX_PATH, "layer_norm");
    if (!f || !launch_1d(f, params, rows, 1)) {
      return 9;
    }
    std::vector<float> output(input.size());
    cuMemcpyDtoH(output.data(), d_out, bytes);
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "embedding_lookup") {
    if (argc != 7) {
      return 10;
    }
    auto weight = read_floats(argv[2]);
    auto indices = read_ints(argv[3]);
    const int count = std::stoi(argv[4]);
    const int dim = std::stoi(argv[5]);
    const std::size_t out_count =
        static_cast<std::size_t>(count) * static_cast<std::size_t>(dim);
    CUdeviceptr d_weight =
        alloc_copy(weight.data(), weight.size() * sizeof(float));
    CUdeviceptr d_indices =
        alloc_copy(indices.data(), indices.size() * sizeof(int));
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, out_count * sizeof(float));
    int c = count;
    int d = dim;
    std::vector<void *> params = {&d_weight, &d_indices, &d_out, &c, &d};
    CUfunction f = load_function(EMBEDDING_PTX_PATH, "embedding_lookup");
    if (!f || !launch_1d(f, params, count)) {
      return 11;
    }
    std::vector<float> output(out_count);
    cuMemcpyDtoH(output.data(), d_out, out_count * sizeof(float));
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "transpose") {
    if (argc != 6) {
      return 12;
    }
    auto input = read_floats(argv[2]);
    const int rows = std::stoi(argv[3]);
    const int cols = std::stoi(argv[4]);
    const std::size_t out_count =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    CUdeviceptr d_in = alloc_copy(input.data(), input.size() * sizeof(float));
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, out_count * sizeof(float));
    int r = rows;
    int c = cols;
    std::vector<void *> params = {&d_in, &d_out, &r, &c};
    CUfunction f = load_function(TRANSPOSE_PTX_PATH, "transpose");
    if (!f ||
        !launch_2d(f, params, (cols + 31) / 32, (rows + 31) / 32, 32, 32)) {
      return 13;
    }
    std::vector<float> output(out_count);
    cuMemcpyDtoH(output.data(), d_out, out_count * sizeof(float));
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "matmul_naive" || mode == "matmul_tiled") {
    if (argc != 8) {
      return 14;
    }
    auto a = read_floats(argv[2]);
    auto b = read_floats(argv[3]);
    const int m = std::stoi(argv[4]);
    const int n = std::stoi(argv[5]);
    const int k = std::stoi(argv[6]);
    const std::size_t out_count =
        static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
    CUdeviceptr d_a = alloc_copy(a.data(), a.size() * sizeof(float));
    CUdeviceptr d_b = alloc_copy(b.data(), b.size() * sizeof(float));
    CUdeviceptr d_c = 0;
    cuMemAlloc(&d_c, out_count * sizeof(float));
    int mm = m;
    int nn = n;
    int kk = k;
    std::vector<void *> params = {&d_a, &d_b, &d_c, &mm, &nn, &kk};
    const std::string name =
        mode == "matmul_naive" ? "matmul_naive" : "matmul_tiled";
    const std::string ptx =
        mode == "matmul_naive" ? MATMUL_NAIVE_PTX_PATH : MATMUL_TILED_PTX_PATH;
    CUfunction f = load_function(ptx, name);
    const bool ok =
        mode == "matmul_naive"
            ? launch_2d(f, params, (n + 15) / 16, (m + 15) / 16, 16, 16)
            : launch_2d(f, params, (n + 15) / 16, (m + 15) / 16, 16, 16);
    if (!f || !ok) {
      return 15;
    }
    std::vector<float> output(out_count);
    cuMemcpyDtoH(output.data(), d_c, out_count * sizeof(float));
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "softmax_row") {
    if (argc != 6) {
      return 16;
    }
    auto input = read_floats(argv[2]);
    const int rows = std::stoi(argv[3]);
    const int cols = std::stoi(argv[4]);
    const std::size_t bytes = input.size() * sizeof(float);
    CUdeviceptr d_in = alloc_copy(input.data(), bytes);
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, bytes);
    int r = rows;
    int c = cols;
    std::vector<void *> params = {&d_in, &d_out, &r, &c};
    CUfunction f = load_function(SOFTMAX_PTX_PATH, "softmax_row");
    if (!f || !launch_1d(f, params, rows, 1)) {
      return 17;
    }
    std::vector<float> output(input.size());
    cuMemcpyDtoH(output.data(), d_out, bytes);
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "reduce_sum") {
    if (argc != 6) {
      return 18;
    }
    auto input = read_floats(argv[2]);
    const int n = std::stoi(argv[3]);
    const int block = std::stoi(argv[4]);
    const std::size_t bytes = input.size() * sizeof(float);
    CUdeviceptr d_in = alloc_copy(input.data(), bytes);
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, sizeof(float));
    int count = n;
    std::vector<void *> params = {&d_in, &d_out, &count};
    CUfunction f = load_function(REDUCE_SUM_PTX_PATH, "reduce_sum");
    if (!f || !launch_1d(f, params, n, block)) {
      return 19;
    }
    float output = 0.0F;
    cuMemcpyDtoH(&output, d_out, sizeof(output));
    std::vector<float> result{output};
    write_floats(out_path, result);
    return 0;
  }

  if (mode == "conv2d_naive" || mode == "conv2d_im2col" ||
      mode == "conv2d_shared") {
    if (argc != 13) {
      return 20;
    }
    auto input = read_floats(argv[2]);
    auto weight = read_floats(argv[3]);
    const int c = std::stoi(argv[4]);
    const int h = std::stoi(argv[5]);
    const int w = std::stoi(argv[6]);
    const int out_c = std::stoi(argv[7]);
    const int kh = std::stoi(argv[8]);
    const int kw = std::stoi(argv[9]);
    const int out_h = std::stoi(argv[10]);
    const int out_w = std::stoi(argv[11]);
    const std::size_t out_count = static_cast<std::size_t>(out_c) *
                                  static_cast<std::size_t>(out_h) *
                                  static_cast<std::size_t>(out_w);
    CUdeviceptr d_in = alloc_copy(input.data(), input.size() * sizeof(float));
    CUdeviceptr d_weight =
        alloc_copy(weight.data(), weight.size() * sizeof(float));
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, out_count * sizeof(float));
    int cc = c;
    int hh = h;
    int ww = w;
    int oc = out_c;
    int kkh = kh;
    int kkw = kw;
    int oh = out_h;
    int ow = out_w;
    std::vector<void *> params;
    if (mode == "conv2d_naive") {
      int batch = 1;
      params = {&d_in, &d_weight, &d_out, &batch, &cc, &hh,
                &ww,   &oc,       &kkh,   &kkw,   &oh, &ow};
    } else {
      params = {&d_in, &d_weight, &d_out, &cc, &hh, &ww,
                &oc,   &kkh,      &kkw,   &oh, &ow};
    }
    const std::string name =
        mode == "conv2d_naive"
            ? "conv2d_naive"
            : (mode == "conv2d_im2col" ? "conv2d_im2col" : "conv2d_shared");
    const std::string ptx =
        mode == "conv2d_naive"
            ? CONV2D_NAIVE_PTX_PATH
            : (mode == "conv2d_im2col" ? CONV2D_IM2COL_PTX_PATH
                                       : CONV2D_SHARED_PTX_PATH);
    CUfunction f = load_function(ptx, name);
    const bool ok =
        mode == "conv2d_shared"
            ? launch_2d(f, params, (out_w + 15) / 16, (out_h + 15) / 16, 16, 16)
            : launch_2d(f, params, (out_w + 15) / 16, (out_h + 15) / 16, 16,
                        16);
    if (!f || !ok) {
      return 21;
    }
    std::vector<float> output(out_count);
    cuMemcpyDtoH(output.data(), d_out, out_count * sizeof(float));
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "attention_scores") {
    if (argc != 9) {
      return 22;
    }
    auto query = read_floats(argv[2]);
    auto key = read_floats(argv[3]);
    const int batch = std::stoi(argv[4]);
    const int heads = std::stoi(argv[5]);
    const int seq = std::stoi(argv[6]);
    const int dim = std::stoi(argv[7]);
    const std::size_t out_count =
        static_cast<std::size_t>(batch) * static_cast<std::size_t>(heads) *
        static_cast<std::size_t>(seq) * static_cast<std::size_t>(seq);
    CUdeviceptr d_q = alloc_copy(query.data(), query.size() * sizeof(float));
    CUdeviceptr d_k = alloc_copy(key.data(), key.size() * sizeof(float));
    CUdeviceptr d_scores = 0;
    cuMemAlloc(&d_scores, out_count * sizeof(float));
    int b = batch;
    int h = heads;
    int s = seq;
    int d = dim;
    std::vector<void *> params = {&d_q, &d_k, &d_scores, &b, &h, &s, &d};
    CUfunction f = load_function(ATTENTION_PTX_PATH, "attention_scores");
    if (!f || !launch_2d(f, params, (seq + 15) / 16, heads, 16, 1)) {
      return 23;
    }
    std::vector<float> output(out_count);
    cuMemcpyDtoH(output.data(), d_scores, out_count * sizeof(float));
    write_floats(out_path, output);
    return 0;
  }

  return 24;
}
