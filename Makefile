BUILD_DIR ?= build
CMAKE_BUILD_TYPE ?= Debug
CMAKE ?= cmake
NINJA ?= ninja
CTEST ?= ctest
PYTHON ?= python3
VENV ?= .venv
VENV_PIP ?= $(VENV)/bin/pip
VENV_BLACK ?= $(VENV)/bin/black
VENV_PRE_COMMIT ?= $(VENV)/bin/pre-commit

CMAKE_FLAGS := \
	-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
	-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
	-DCUDA_SIM_ENABLE_JIT=OFF \
	-DCUDA_SIM_ENABLE_SANITIZERS=OFF

.PHONY: all build configure test run clean help install-hooks precommit format format-python venv compile_commands

all: build

build: configure
	$(CMAKE) --build $(BUILD_DIR)

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -G Ninja $(CMAKE_FLAGS)

compile_commands: configure
	ln -sfn $(BUILD_DIR)/compile_commands.json compile_commands.json

test: build
	$(CTEST) --test-dir $(BUILD_DIR) --output-on-failure

run: build
	./$(BUILD_DIR)/src/cli/cuda_sim_run

venv:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install -e '.[dev]'

install-hooks: venv
	git config --unset-all core.hooksPath || true
	$(VENV_PRE_COMMIT) install

precommit:
	$(VENV_PRE_COMMIT) run --all-files

format:
	find src tests -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \) -print0 | xargs -0 clang-format -i

format-python:
	$(VENV_BLACK) tests

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo "Targets:"
	@echo "  make build      Configure with Ninja and build"
	@echo "  make test       Build and run CTest"
	@echo "  make run        Build and run the CLI smoke executable"
	@echo "  make compile_commands  Generate root compile_commands.json"
	@echo "  make venv          Create .venv and install dev tools"
	@echo "  make install-hooks  Install pre-commit Git hooks"
	@echo "  make precommit      Run pre-commit on all files"
	@echo "  make format     Format C++ sources with clang-format"
	@echo "  make format-python  Format Python sources with black from .venv"
	@echo "  make clean      Remove the build directory"
