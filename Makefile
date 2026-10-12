BUILD_DIR ?= build
CMAKE_BUILD_TYPE ?= Debug
CMAKE ?= cmake
CTEST ?= ctest
PYTHON ?= python3
VENV ?= .venv
VENV_PIP ?= $(VENV)/bin/pip
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
TRITON_VERSION ?= 3.8.0

CMAKE_FLAGS := \
	-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
	-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
	-DCUDA_SIM_ENABLE_JIT=OFF \
	-DCUDA_SIM_ENABLE_SANITIZERS=OFF

.PHONY: all build configure test run clean help precommit format format-python venv compile_commands

all: build

$(VENV_MARKER):
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install "torch==$(TORCH_VERSION)" "triton==$(TRITON_VERSION)" --index-url $(PYTORCH_INDEX_URL)
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

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo "Targets:"
	@echo "  make build      Configure with Ninja and build"
	@echo "  make test       Build and run CTest"
	@echo "  make run        Build and run the CLI smoke executable"
	@echo "  make compile_commands  Generate root compile_commands.json"
	@echo "  make venv          Bootstrap .venv and install all dependencies"
	@echo "  make precommit      Run pre-commit on all files"
	@echo "  make format     Format C++ sources with clang-format"
	@echo "  make format-python  Format Python sources with black from .venv"
	@echo "  make clean      Remove the build directory"
