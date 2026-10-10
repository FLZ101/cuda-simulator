BUILD_DIR ?= build
CMAKE_BUILD_TYPE ?= Debug
CMAKE ?= cmake
NINJA ?= ninja
CTEST ?= ctest

CMAKE_FLAGS := \
	-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
	-DCUDA_SIM_ENABLE_JIT=OFF \
	-DCUDA_SIM_ENABLE_SANITIZERS=OFF

.PHONY: all build configure test run clean help install-hooks format

all: build

build: configure
	$(CMAKE) --build $(BUILD_DIR)

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -G Ninja $(CMAKE_FLAGS)

test: build
	$(CTEST) --test-dir $(BUILD_DIR) --output-on-failure

run: build
	./$(BUILD_DIR)/src/cli/cuda_sim_run

install-hooks:
	git config core.hooksPath .githooks

format:
	find src tests -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \) -print0 | xargs -0 clang-format -i

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo "Targets:"
	@echo "  make build      Configure with Ninja and build"
	@echo "  make test       Build and run CTest"
	@echo "  make run        Build and run the CLI smoke executable"
	@echo "  make install-hooks  Configure Git to use .githooks"
	@echo "  make format     Format C++ sources with clang-format"
	@echo "  make clean      Remove the build directory"
