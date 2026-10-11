#include <cuda.h>

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

bool initialize_cuda() {
  if (cuInit(0) != CUDA_SUCCESS) {
    return false;
  }
  CUdevice device = 0;
  if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) {
    return false;
  }
  CUcontext context = nullptr;
  if (cuDevicePrimaryCtxRetain(&context, device) != CUDA_SUCCESS) {
    return false;
  }
  return cuCtxSetCurrent(context) == CUDA_SUCCESS;
}

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

  CUmodule module = nullptr;
  if (cuModuleLoadData(&module, ptx.c_str()) != CUDA_SUCCESS) {
    return nullptr;
  }
  // The shim stores the CUfunction as a pointer to the name. Keep a stable
  // string alive until cuLaunchKernel has consumed it.
  static std::vector<std::string> function_names;
  function_names.push_back(name);
  const std::string &stable_name = function_names.back();
  CUfunction function = nullptr;
  if (cuModuleGetFunction(&function, module, stable_name.c_str()) !=
      CUDA_SUCCESS) {
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

bool launch_1d_shared(CUfunction function, const std::vector<void *> &params,
                      int n, int block, std::size_t shared_mem) {
  const unsigned int blocks =
      (static_cast<unsigned int>(n) + static_cast<unsigned int>(block) - 1) /
      static_cast<unsigned int>(block);
  return cuLaunchKernel(function, blocks, 1, 1,
                        static_cast<unsigned int>(block), 1, 1, shared_mem,
                        nullptr, const_cast<void **>(params.data()),
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
  if (!initialize_cuda()) {
    return 25;
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
    const bool ok = launch_2d(f, params, (n + 15) / 16, (m + 15) / 16, 16, 16);
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
    if (!f ||
        !launch_1d_shared(f, params, n, block,
                          static_cast<std::size_t>(block) * sizeof(float))) {
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

  if (mode == "copy") {
    if (argc != 5) {
      return 25;
    }
    auto input = read_floats(argv[2]);
    const int n = std::stoi(argv[3]);
    const std::size_t bytes = input.size() * sizeof(float);
    CUdeviceptr d_in = alloc_copy(input.data(), bytes);
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, bytes);
    int count = n;
    std::vector<void *> params = {&d_in, &d_out, &count};
    CUfunction f = load_function(COPY_PTX_PATH, "copy_kernel");
    if (!f || !launch_1d(f, params, n)) {
      return 26;
    }
    std::vector<float> output(input.size());
    cuMemcpyDtoH(output.data(), d_out, bytes);
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "saxpy") {
    if (argc != 7) {
      return 27;
    }
    auto x = read_floats(argv[2]);
    auto y = read_floats(argv[3]);
    const float alpha = std::stof(argv[4]);
    const int n = std::stoi(argv[5]);
    const std::size_t bytes = x.size() * sizeof(float);
    CUdeviceptr d_x = alloc_copy(x.data(), bytes);
    CUdeviceptr d_y = alloc_copy(y.data(), y.size() * sizeof(float));
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, bytes);
    int count = n;
    float a = alpha;
    std::vector<void *> params = {&d_x, &d_y, &d_out, &a, &count};
    CUfunction f = load_function(SAXPY_PTX_PATH, "saxpy");
    if (!f || !launch_1d(f, params, n)) {
      return 28;
    }
    std::vector<float> output(x.size());
    cuMemcpyDtoH(output.data(), d_out, bytes);
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "relu") {
    if (argc != 5) {
      return 29;
    }
    auto input = read_floats(argv[2]);
    const int n = std::stoi(argv[3]);
    const std::size_t bytes = input.size() * sizeof(float);
    CUdeviceptr d_in = alloc_copy(input.data(), bytes);
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, bytes);
    int count = n;
    std::vector<void *> params = {&d_in, &d_out, &count};
    CUfunction f = load_function(RELU_PTX_PATH, "relu");
    if (!f || !launch_1d(f, params, n)) {
      return 30;
    }
    std::vector<float> output(input.size());
    cuMemcpyDtoH(output.data(), d_out, bytes);
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "elementwise_add" || mode == "elementwise_mul") {
    if (argc != 6) {
      return 31;
    }
    auto a = read_floats(argv[2]);
    auto b = read_floats(argv[3]);
    const int n = std::stoi(argv[4]);
    const std::size_t bytes = a.size() * sizeof(float);
    CUdeviceptr d_a = alloc_copy(a.data(), bytes);
    CUdeviceptr d_b = alloc_copy(b.data(), b.size() * sizeof(float));
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, bytes);
    int count = n;
    std::vector<void *> params = {&d_a, &d_b, &d_out, &count};
    const std::string ptx = mode == "elementwise_add"
                                ? ELEMENTWISE_ADD_PTX_PATH
                                : ELEMENTWISE_MUL_PTX_PATH;
    const std::string name =
        mode == "elementwise_add" ? "elementwise_add" : "elementwise_mul";
    CUfunction f = load_function(ptx, name);
    if (!f || !launch_1d(f, params, n)) {
      return 32;
    }
    std::vector<float> output(a.size());
    cuMemcpyDtoH(output.data(), d_out, bytes);
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "scale") {
    if (argc != 6) {
      return 33;
    }
    auto input = read_floats(argv[2]);
    const float factor = std::stof(argv[3]);
    const int n = std::stoi(argv[4]);
    const std::size_t bytes = input.size() * sizeof(float);
    CUdeviceptr d_in = alloc_copy(input.data(), bytes);
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, bytes);
    int count = n;
    float f = factor;
    std::vector<void *> params = {&d_in, &d_out, &f, &count};
    CUfunction func = load_function(SCALE_PTX_PATH, "scale");
    if (!func || !launch_1d(func, params, n)) {
      return 34;
    }
    std::vector<float> output(input.size());
    cuMemcpyDtoH(output.data(), d_out, bytes);
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "mean") {
    if (argc != 5) {
      return 35;
    }
    auto input = read_floats(argv[2]);
    const int n = std::stoi(argv[3]);
    const std::size_t bytes = input.size() * sizeof(float);
    CUdeviceptr d_in = alloc_copy(input.data(), bytes);
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, sizeof(float));
    int count = n;
    std::vector<void *> params = {&d_in, &d_out, &count};
    CUfunction f = load_function(MEAN_PTX_PATH, "mean");
    if (!f || !launch_1d(f, params, 1, 1)) {
      return 36;
    }
    float output = 0.0F;
    cuMemcpyDtoH(&output, d_out, sizeof(output));
    std::vector<float> result{output};
    write_floats(out_path, result);
    return 0;
  }

  if (mode == "dropout") {
    if (argc != 6) {
      return 37;
    }
    auto input = read_floats(argv[2]);
    auto mask = read_floats(argv[3]);
    const int n = std::stoi(argv[4]);
    const std::size_t bytes = input.size() * sizeof(float);
    CUdeviceptr d_in = alloc_copy(input.data(), bytes);
    CUdeviceptr d_mask = alloc_copy(mask.data(), mask.size() * sizeof(float));
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, bytes);
    int count = n;
    std::vector<void *> params = {&d_in, &d_mask, &d_out, &count};
    CUfunction f = load_function(DROPOUT_PTX_PATH, "dropout");
    if (!f || !launch_1d(f, params, n)) {
      return 38;
    }
    std::vector<float> output(input.size());
    cuMemcpyDtoH(output.data(), d_out, bytes);
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "causal_mask") {
    if (argc != 5) {
      return 39;
    }
    auto input = read_floats(argv[2]);
    const int seq_len = std::stoi(argv[3]);
    const std::size_t bytes = input.size() * sizeof(float);
    CUdeviceptr d_scores = alloc_copy(input.data(), bytes);
    int s = seq_len;
    std::vector<void *> params = {&d_scores, &s};
    CUfunction f = load_function(CAUSAL_MASK_PTX_PATH, "causal_mask");
    if (!f || !launch_2d(f, params, (seq_len + 255) / 256, seq_len, 256, 1)) {
      return 40;
    }
    std::vector<float> output(input.size());
    cuMemcpyDtoH(output.data(), d_scores, bytes);
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "residual_add") {
    if (argc != 6) {
      return 41;
    }
    auto input = read_floats(argv[2]);
    auto residual = read_floats(argv[3]);
    const int n = std::stoi(argv[4]);
    const std::size_t bytes = input.size() * sizeof(float);
    CUdeviceptr d_in = alloc_copy(input.data(), bytes);
    CUdeviceptr d_res =
        alloc_copy(residual.data(), residual.size() * sizeof(float));
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, bytes);
    int count = n;
    std::vector<void *> params = {&d_in, &d_res, &d_out, &count};
    CUfunction f = load_function(RESIDUAL_ADD_PTX_PATH, "residual_add");
    if (!f || !launch_1d(f, params, n)) {
      return 42;
    }
    std::vector<float> output(input.size());
    cuMemcpyDtoH(output.data(), d_out, bytes);
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "rms_norm") {
    if (argc != 7) {
      return 43;
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
    CUfunction f = load_function(RMS_NORM_PTX_PATH, "rms_norm");
    if (!f || !launch_1d(f, params, rows, 1)) {
      return 44;
    }
    std::vector<float> output(input.size());
    cuMemcpyDtoH(output.data(), d_out, bytes);
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "concat") {
    if (argc != 7) {
      return 45;
    }
    auto a = read_floats(argv[2]);
    auto b = read_floats(argv[3]);
    const int a_size = std::stoi(argv[4]);
    const int b_size = std::stoi(argv[5]);
    const std::size_t bytes = (a.size() + b.size()) * sizeof(float);
    CUdeviceptr d_a = alloc_copy(a.data(), a.size() * sizeof(float));
    CUdeviceptr d_b = alloc_copy(b.data(), b.size() * sizeof(float));
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, bytes);
    int as = a_size;
    int bs = b_size;
    std::vector<void *> params = {&d_a, &d_b, &d_out, &as, &bs};
    CUfunction f = load_function(CONCAT_PTX_PATH, "concat");
    if (!f || !launch_1d(f, params, a_size + b_size)) {
      return 46;
    }
    std::vector<float> output(a.size() + b.size());
    cuMemcpyDtoH(output.data(), d_out, bytes);
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "split") {
    if (argc != 6) {
      return 47;
    }
    auto input = read_floats(argv[2]);
    const int split = std::stoi(argv[3]);
    const int n = std::stoi(argv[4]);
    const std::size_t in_bytes = input.size() * sizeof(float);
    const std::size_t a_bytes = static_cast<std::size_t>(split) * sizeof(float);
    const std::size_t b_bytes =
        static_cast<std::size_t>(n - split) * sizeof(float);
    CUdeviceptr d_in = alloc_copy(input.data(), in_bytes);
    CUdeviceptr d_a = 0;
    CUdeviceptr d_b = 0;
    cuMemAlloc(&d_a, a_bytes);
    cuMemAlloc(&d_b, b_bytes);
    int s = split;
    int count = n;
    std::vector<void *> params = {&d_in, &d_a, &d_b, &s, &count};
    CUfunction f = load_function(SPLIT_PTX_PATH, "split");
    if (!f || !launch_1d(f, params, n)) {
      return 48;
    }
    std::vector<float> a(split);
    std::vector<float> b(n - split);
    cuMemcpyDtoH(a.data(), d_a, a_bytes);
    cuMemcpyDtoH(b.data(), d_b, b_bytes);
    a.insert(a.end(), b.begin(), b.end());
    write_floats(out_path, a);
    return 0;
  }

  if (mode == "argmax") {
    if (argc != 5) {
      return 49;
    }
    auto input = read_floats(argv[2]);
    const int n = std::stoi(argv[3]);
    CUdeviceptr d_in = alloc_copy(input.data(), input.size() * sizeof(float));
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, sizeof(float));
    int count = n;
    std::vector<void *> params = {&d_in, &d_out, &count};
    CUfunction f = load_function(ARGMAX_PTX_PATH, "argmax");
    if (!f || !launch_1d(f, params, 1, 1)) {
      return 50;
    }
    float output = 0.0F;
    cuMemcpyDtoH(&output, d_out, sizeof(output));
    std::vector<float> result{output};
    write_floats(out_path, result);
    return 0;
  }

  if (mode == "clamp") {
    if (argc != 7) {
      return 51;
    }
    auto input = read_floats(argv[2]);
    const float min_value = std::stof(argv[3]);
    const float max_value = std::stof(argv[4]);
    const int n = std::stoi(argv[5]);
    const std::size_t bytes = input.size() * sizeof(float);
    CUdeviceptr d_in = alloc_copy(input.data(), bytes);
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, bytes);
    int count = n;
    float mn = min_value;
    float mx = max_value;
    std::vector<void *> params = {&d_in, &d_out, &mn, &mx, &count};
    CUfunction f = load_function(CLAMP_PTX_PATH, "clamp");
    if (!f || !launch_1d(f, params, n)) {
      return 52;
    }
    std::vector<float> output(input.size());
    cuMemcpyDtoH(output.data(), d_out, bytes);
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "neg" || mode == "square") {
    if (argc != 5) {
      return 53;
    }
    auto input = read_floats(argv[2]);
    const int n = std::stoi(argv[3]);
    const std::size_t bytes = input.size() * sizeof(float);
    CUdeviceptr d_in = alloc_copy(input.data(), bytes);
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, bytes);
    int count = n;
    std::vector<void *> params = {&d_in, &d_out, &count};
    const std::string ptx = mode == "neg" ? NEG_PTX_PATH : SQUARE_PTX_PATH;
    const std::string name = mode == "neg" ? "neg" : "square";
    CUfunction f = load_function(ptx, name);
    if (!f || !launch_1d(f, params, n)) {
      return 54;
    }
    std::vector<float> output(input.size());
    cuMemcpyDtoH(output.data(), d_out, bytes);
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "log_softmax") {
    if (argc != 6) {
      return 55;
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
    CUfunction f = load_function(LOG_SOFTMAX_PTX_PATH, "log_softmax");
    if (!f || !launch_1d(f, params, rows, 1)) {
      return 56;
    }
    std::vector<float> output(input.size());
    cuMemcpyDtoH(output.data(), d_out, bytes);
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "cross_entropy") {
    if (argc != 7) {
      return 57;
    }
    auto logits = read_floats(argv[2]);
    auto targets = read_ints(argv[3]);
    const int batch = std::stoi(argv[4]);
    const int classes = std::stoi(argv[5]);
    const std::size_t logits_bytes = logits.size() * sizeof(float);
    CUdeviceptr d_logits = alloc_copy(logits.data(), logits_bytes);
    CUdeviceptr d_targets =
        alloc_copy(targets.data(), targets.size() * sizeof(int));
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, static_cast<std::size_t>(batch) * sizeof(float));
    int b = batch;
    int c = classes;
    std::vector<void *> params = {&d_logits, &d_targets, &d_out, &b, &c};
    CUfunction f = load_function(CROSS_ENTROPY_PTX_PATH, "cross_entropy");
    if (!f || !launch_1d(f, params, batch, 1)) {
      return 58;
    }
    std::vector<float> output(static_cast<std::size_t>(batch));
    cuMemcpyDtoH(output.data(), d_out, output.size() * sizeof(float));
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "sigmoid" || mode == "tanh" || mode == "silu" ||
      mode == "reciprocal" || mode == "exp" || mode == "log" ||
      mode == "sqrt" || mode == "rsqrt" || mode == "abs" || mode == "relu6" ||
      mode == "softplus" || mode == "exp2" || mode == "log2" ||
      mode == "floor" || mode == "ceil" || mode == "rint" || mode == "trunc" ||
      mode == "acos" || mode == "erfc" || mode == "hard_sigmoid" ||
      mode == "hardswish" || mode == "softsign" || mode == "log_sigmoid" ||
      mode == "cosh" || mode == "sinh" || mode == "asin" || mode == "atan" ||
      mode == "cbrt" || mode == "mish" || mode == "expm1") {
    if (argc != 5) {
      return 59;
    }
    auto input = read_floats(argv[2]);
    const int n = std::stoi(argv[3]);
    const std::size_t bytes = input.size() * sizeof(float);
    CUdeviceptr d_in = alloc_copy(input.data(), bytes);
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, bytes);
    int count = n;
    std::vector<void *> params = {&d_in, &d_out, &count};
    std::string ptx;
    std::string name;
    if (mode == "sigmoid") {
      ptx = SIGMOID_PTX_PATH;
      name = "sigmoid";
    } else if (mode == "tanh") {
      ptx = TANH_PTX_PATH;
      name = "tanh_kernel";
    } else if (mode == "silu") {
      ptx = SILU_PTX_PATH;
      name = "silu";
    } else if (mode == "exp") {
      ptx = EXP_PTX_PATH;
      name = "exp_kernel";
    } else if (mode == "log") {
      ptx = LOG_PTX_PATH;
      name = "log_kernel";
    } else if (mode == "sqrt") {
      ptx = SQRT_PTX_PATH;
      name = "sqrt_kernel";
    } else if (mode == "rsqrt") {
      ptx = RSQRT_PTX_PATH;
      name = "rsqrt_kernel";
    } else if (mode == "abs") {
      ptx = ABS_PTX_PATH;
      name = "abs_kernel";
    } else if (mode == "relu6") {
      ptx = RELU6_PTX_PATH;
      name = "relu6";
    } else if (mode == "softplus") {
      ptx = SOFTPLUS_PTX_PATH;
      name = "softplus";
    } else if (mode == "exp2") {
      ptx = EXP2_PTX_PATH;
      name = "exp2_kernel";
    } else if (mode == "log2") {
      ptx = LOG2_PTX_PATH;
      name = "log2_kernel";
    } else if (mode == "floor") {
      ptx = FLOOR_PTX_PATH;
      name = "floor_kernel";
    } else if (mode == "ceil") {
      ptx = CEIL_PTX_PATH;
      name = "ceil_kernel";
    } else if (mode == "rint") {
      ptx = RINT_PTX_PATH;
      name = "rint_kernel";
    } else if (mode == "trunc") {
      ptx = TRUNC_PTX_PATH;
      name = "trunc_kernel";
    } else if (mode == "acos") {
      ptx = ACOS_PTX_PATH;
      name = "acos_kernel";
    } else if (mode == "erfc") {
      ptx = ERFC_PTX_PATH;
      name = "erfc_kernel";
    } else if (mode == "hard_sigmoid") {
      ptx = HARD_SIGMOID_PTX_PATH;
      name = "hard_sigmoid";
    } else if (mode == "hardswish") {
      ptx = HARDSWISH_PTX_PATH;
      name = "hardswish";
    } else if (mode == "softsign") {
      ptx = SOFTSIGN_PTX_PATH;
      name = "softsign";
    } else if (mode == "log_sigmoid") {
      ptx = LOG_SIGMOID_PTX_PATH;
      name = "log_sigmoid";
    } else if (mode == "cosh") {
      ptx = COSH_PTX_PATH;
      name = "cosh_kernel";
    } else if (mode == "sinh") {
      ptx = SINH_PTX_PATH;
      name = "sinh_kernel";
    } else if (mode == "asin") {
      ptx = ASIN_PTX_PATH;
      name = "asin_kernel";
    } else if (mode == "atan") {
      ptx = ATAN_PTX_PATH;
      name = "atan_kernel";
    } else if (mode == "cbrt") {
      ptx = CBRT_PTX_PATH;
      name = "cbrt_kernel";
    } else if (mode == "mish") {
      ptx = MISH_PTX_PATH;
      name = "mish_kernel";
    } else if (mode == "expm1") {
      ptx = EXPM1_PTX_PATH;
      name = "expm1_kernel";
    } else {
      ptx = RECIPROCAL_PTX_PATH;
      name = "reciprocal";
    }
    CUfunction f = load_function(ptx, name);
    if (!f || !launch_1d(f, params, n)) {
      return 60;
    }
    std::vector<float> output(input.size());
    cuMemcpyDtoH(output.data(), d_out, bytes);
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "pow") {
    if (argc != 6) {
      return 61;
    }
    auto input = read_floats(argv[2]);
    const float exponent = std::stof(argv[3]);
    const int n = std::stoi(argv[4]);
    const std::size_t bytes = input.size() * sizeof(float);
    CUdeviceptr d_in = alloc_copy(input.data(), bytes);
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, bytes);
    int count = n;
    float e = exponent;
    std::vector<void *> params = {&d_in, &d_out, &e, &count};
    CUfunction f = load_function(POW_PTX_PATH, "pow_kernel");
    if (!f || !launch_1d(f, params, n)) {
      return 62;
    }
    std::vector<float> output(input.size());
    cuMemcpyDtoH(output.data(), d_out, bytes);
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "minimum" || mode == "maximum") {
    if (argc != 6) {
      return 63;
    }
    auto a = read_floats(argv[2]);
    auto b = read_floats(argv[3]);
    const int n = std::stoi(argv[4]);
    const std::size_t bytes = a.size() * sizeof(float);
    CUdeviceptr d_a = alloc_copy(a.data(), bytes);
    CUdeviceptr d_b = alloc_copy(b.data(), b.size() * sizeof(float));
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, bytes);
    int count = n;
    std::vector<void *> params = {&d_a, &d_b, &d_out, &count};
    const std::string ptx =
        mode == "minimum" ? MINIMUM_PTX_PATH : MAXIMUM_PTX_PATH;
    const std::string name = mode == "minimum" ? "min_kernel" : "max_kernel";
    CUfunction f = load_function(ptx, name);
    if (!f || !launch_1d(f, params, n)) {
      return 64;
    }
    std::vector<float> output(a.size());
    cuMemcpyDtoH(output.data(), d_out, bytes);
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "subtract" || mode == "divide" || mode == "copysign" ||
      mode == "lerp" || mode == "atan2" || mode == "hypot") {
    if (argc != 6) {
      return 71;
    }
    auto a = read_floats(argv[2]);
    auto b = read_floats(argv[3]);
    const int n = std::stoi(argv[4]);
    const std::size_t bytes = a.size() * sizeof(float);
    CUdeviceptr d_a = alloc_copy(a.data(), bytes);
    CUdeviceptr d_b = alloc_copy(b.data(), b.size() * sizeof(float));
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, bytes);
    int count = n;
    std::vector<void *> params = {&d_a, &d_b, &d_out, &count};
    std::string ptx;
    std::string name;
    if (mode == "subtract") {
      ptx = SUBTRACT_PTX_PATH;
      name = "subtract_kernel";
    } else if (mode == "divide") {
      ptx = DIVIDE_PTX_PATH;
      name = "divide_kernel";
    } else if (mode == "copysign") {
      ptx = COPYSIGN_PTX_PATH;
      name = "copysign_kernel";
    } else if (mode == "lerp") {
      ptx = LERP_PTX_PATH;
      name = "lerp_kernel";
    } else if (mode == "atan2") {
      ptx = ATAN2_PTX_PATH;
      name = "atan2_kernel";
    } else {
      ptx = HYPOT_PTX_PATH;
      name = "hypot_kernel";
    }
    CUfunction f = load_function(ptx, name);
    if (!f || !launch_1d(f, params, n)) {
      return 72;
    }
    std::vector<float> output(a.size());
    cuMemcpyDtoH(output.data(), d_out, bytes);
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "leaky_relu") {
    if (argc != 6) {
      return 65;
    }
    auto input = read_floats(argv[2]);
    const float alpha = std::stof(argv[3]);
    const int n = std::stoi(argv[4]);
    const std::size_t bytes = input.size() * sizeof(float);
    CUdeviceptr d_in = alloc_copy(input.data(), bytes);
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, bytes);
    int count = n;
    float a = alpha;
    std::vector<void *> params = {&d_in, &d_out, &a, &count};
    CUfunction f = load_function(LEAKY_RELU_PTX_PATH, "leaky_relu");
    if (!f || !launch_1d(f, params, n)) {
      return 66;
    }
    std::vector<float> output(input.size());
    cuMemcpyDtoH(output.data(), d_out, bytes);
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "swiglu") {
    if (argc != 6) {
      return 67;
    }
    auto a = read_floats(argv[2]);
    auto b = read_floats(argv[3]);
    const int n = std::stoi(argv[4]);
    const std::size_t bytes = a.size() * sizeof(float);
    CUdeviceptr d_a = alloc_copy(a.data(), bytes);
    CUdeviceptr d_b = alloc_copy(b.data(), b.size() * sizeof(float));
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, bytes);
    int count = n;
    std::vector<void *> params = {&d_a, &d_b, &d_out, &count};
    CUfunction f = load_function(SWIGLU_PTX_PATH, "swiglu");
    if (!f || !launch_1d(f, params, n)) {
      return 68;
    }
    std::vector<float> output(a.size());
    cuMemcpyDtoH(output.data(), d_out, bytes);
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "elu") {
    if (argc != 6) {
      return 69;
    }
    auto input = read_floats(argv[2]);
    const float alpha = std::stof(argv[3]);
    const int n = std::stoi(argv[4]);
    const std::size_t bytes = input.size() * sizeof(float);
    CUdeviceptr d_in = alloc_copy(input.data(), bytes);
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, bytes);
    int count = n;
    float a = alpha;
    std::vector<void *> params = {&d_in, &d_out, &a, &count};
    CUfunction f = load_function(ELU_PTX_PATH, "elu");
    if (!f || !launch_1d(f, params, n)) {
      return 70;
    }
    std::vector<float> output(input.size());
    cuMemcpyDtoH(output.data(), d_out, bytes);
    write_floats(out_path, output);
    return 0;
  }

  if (mode == "gelu") {
    if (argc != 5) {
      return 25;
    }
    auto input = read_floats(argv[2]);
    const int n = std::stoi(argv[3]);
    const std::size_t bytes = input.size() * sizeof(float);
    CUdeviceptr d_in = alloc_copy(input.data(), bytes);
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, bytes);
    int count = n;
    std::vector<void *> params = {&d_in, &d_out, &count};
    CUfunction f = load_function(GELU_PTX_PATH, "gelu");
    if (!f || !launch_1d(f, params, n)) {
      return 26;
    }
    std::vector<float> output(input.size());
    cuMemcpyDtoH(output.data(), d_out, bytes);
    write_floats(out_path, output);
    return 0;
  }

  return 24;
}
