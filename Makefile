BUILD_DIR ?= build
CMAKE_BUILD_TYPE ?= Debug
CMAKE ?= cmake
CTEST ?= ctest
PYTHON ?= python3
VENV ?= .venv
VENV_PIP ?= $(VENV)/bin/pip
VENV_PYTHON ?= $(VENV)/bin/python
VENV_BLACK ?= $(VENV)/bin/black
VENV_PRE_COMMIT ?= $(VENV)/bin/pre-commit
VENV_MARKER ?= $(VENV)/.marker

# Detect the CUDA toolkit version and derive the PyTorch wheel tag. PyTorch
# publishes wheels per CUDA release line (CUDA 13 -> cu130), not per toolkit
# minor/patch. Override PYTORCH_CUDA_TAG for a different CUDA line.
NVCC ?= $(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
CUDA_MAJOR := $(shell $(NVCC) --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\.[0-9]*.*/\1/p')
PYTORCH_CUDA_TAG ?= cu$(CUDA_MAJOR)0
PYTORCH_INDEX_URL ?= https://download.pytorch.org/whl/$(PYTORCH_CUDA_TAG)
TORCH_VERSION ?= 2.14.1+$(PYTORCH_CUDA_TAG)
TORCHVISION_VERSION ?= 0.29.1+$(PYTORCH_CUDA_TAG)
TRITON_VERSION ?= 3.8.0

RESNET_DIR ?= $(BUILD_DIR)/models/resnet50
RESNET_RUNNER_SRC ?= $(RESNET_DIR)/resnet50_runner.cu
RESNET_RUNNER ?= $(RESNET_DIR)/resnet50_runner
RESNET_MODEL_FILE ?= tests/models/fixtures/resnet50.py
RESNET_BATCH ?= 1
RESNET_IN_CHANNELS ?= 3
RESNET_HEIGHT ?= 4 # 224
RESNET_WIDTH ?= 4 # 224
RESNET_BLOCK ?= 1 # 32
RESNET_ARCH ?= 86
CUDA_SIM_EXEC_THREADS ?= auto

CMAKE_FLAGS := \
	-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
	-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
	-DCUDA_SIM_ENABLE_JIT=OFF \
	-DCUDA_SIM_ENABLE_SANITIZERS=OFF

.PHONY: all build configure test run clean help precommit format format-python venv compile_commands resnet-cpu resnet-gpu resnet-gpu-cudasim

all: build

$(VENV_MARKER):
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install "torch==$(TORCH_VERSION)" "torchvision==$(TORCHVISION_VERSION)" "triton==$(TRITON_VERSION)" --index-url $(PYTORCH_INDEX_URL)
	$(VENV_PIP) install -r requirements-dev.txt
	$(VENV_PRE_COMMIT) install
	touch "$(VENV_MARKER)"

venv: $(VENV_MARKER)

precommit: venv
	$(VENV_PRE_COMMIT) run --all-files

format:
	find src tests -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \) -print0 | xargs -0 clang-format -i

format-python: venv
	$(VENV_BLACK) tests

configure: venv
	$(CMAKE) -S . -B $(BUILD_DIR) -G Ninja $(CMAKE_FLAGS)

compile_commands: configure
	ln -sfn $(BUILD_DIR)/compile_commands.json compile_commands.json

build: configure
	$(CMAKE) --build $(BUILD_DIR)

test: build
	$(CTEST) --test-dir $(BUILD_DIR) --output-on-failure

run: build
	./$(BUILD_DIR)/src/cli/cuda_sim_run

$(RESNET_RUNNER_SRC): venv
	mkdir -p $(RESNET_DIR)
	$(VENV_PYTHON) tests/models/generate_model_runner.py \
		--spec module \
		--module-file $(RESNET_MODEL_FILE) \
		--batch $(RESNET_BATCH) \
		--in-channels $(RESNET_IN_CHANNELS) \
		--height $(RESNET_HEIGHT) \
		--width $(RESNET_WIDTH) \
		--block $(RESNET_BLOCK) \
		--output-cpp $(RESNET_RUNNER_SRC)

$(RESNET_RUNNER): $(RESNET_RUNNER_SRC)
	$(NVCC) -arch=sm_$(RESNET_ARCH) -o $(RESNET_RUNNER) $(RESNET_RUNNER_SRC) -lcuda

resnet-cpu: build $(RESNET_RUNNER)
	$(VENV_PYTHON) tests/models/model_runner.py \
		--spec module \
		--mode cpu \
		--module-file $(RESNET_MODEL_FILE) \
		--batch $(RESNET_BATCH) \
		--in-channels $(RESNET_IN_CHANNELS) \
		--height $(RESNET_HEIGHT) \
		--width $(RESNET_WIDTH) \
		--runner $(RESNET_RUNNER) \
		--cudart $(BUILD_DIR)/src/compat/libcudart_shim.so \
		--cuda $(BUILD_DIR)/src/compat/libcuda_shim.so \
		--core-dir $(BUILD_DIR)/src/core \
		--output-dir $(RESNET_DIR)

resnet-gpu: resnet-cpu
	$(VENV_PYTHON) tests/models/model_runner.py \
		--spec module \
		--mode gpu \
		--compare \
		--module-file $(RESNET_MODEL_FILE) \
		--batch $(RESNET_BATCH) \
		--in-channels $(RESNET_IN_CHANNELS) \
		--height $(RESNET_HEIGHT) \
		--width $(RESNET_WIDTH) \
		--runner $(RESNET_RUNNER) \
		--cudart $(BUILD_DIR)/src/compat/libcudart_shim.so \
		--cuda $(BUILD_DIR)/src/compat/libcuda_shim.so \
		--core-dir $(BUILD_DIR)/src/core \
		--output-dir $(RESNET_DIR)

resnet-gpu-cudasim: resnet-cpu
	CUDA_SIM_EXEC_THREADS=$(CUDA_SIM_EXEC_THREADS) $(VENV_PYTHON) tests/models/model_runner.py \
		--spec module \
		--mode gpu \
		--simulate \
		--compare \
		--module-file $(RESNET_MODEL_FILE) \
		--batch $(RESNET_BATCH) \
		--in-channels $(RESNET_IN_CHANNELS) \
		--height $(RESNET_HEIGHT) \
		--width $(RESNET_WIDTH) \
		--runner $(RESNET_RUNNER) \
		--cudart $(BUILD_DIR)/src/compat/libcudart_shim.so \
		--cuda $(BUILD_DIR)/src/compat/libcuda_shim.so \
		--core-dir $(BUILD_DIR)/src/core \
		--output-dir $(RESNET_DIR)

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo "Targets:"
	@echo "  make build      Configure with Ninja and build"
	@echo "  make test       Build and run CTest"
	@echo "  make run        Build and run the CLI smoke executable"
	@echo "  make resnet-cpu Run the pretrained ResNet-50 PyTorch CPU reference"
	@echo "  make resnet-gpu Run ResNet-50 on the real GPU and compare to CPU"
	@echo "  make resnet-gpu-cudasim Run ResNet-50 through the CUDA simulator"
	@echo "  make compile_commands  Generate root compile_commands.json"
	@echo "  make venv          Bootstrap .venv and install all dependencies"
	@echo "  make precommit      Run pre-commit on all files"
	@echo "  make format     Format C++ sources with clang-format"
	@echo "  make format-python  Format Python sources with black from .venv"
	@echo "  make clean      Remove the build directory"
