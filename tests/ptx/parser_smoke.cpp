#include "ptx/lower.h"
#include "ptx/module.h"

#include <fstream>
#include <sstream>
#include <string>

int main() {
  std::ifstream input(PTX_FIXTURE_PATH);
  if (!input) {
    return 1;
  }

  std::ostringstream buffer;
  buffer << input.rdbuf();

  const cudasim::ptx::Module module = cudasim::ptx::parse_ptx(buffer.str());
  if (module.functions.size() != 1) {
    return 2;
  }
  if (module.functions[0].name != "add_kernel") {
    return 3;
  }

  const cudasim::ptx::ModuleIR ir = cudasim::ptx::lower(module);
  if (ir.functions.size() != 1) {
    return 4;
  }
  if (ir.functions[0].blocks.empty()) {
    return 5;
  }
  if (ir.functions[0].blocks[0].instructions.empty()) {
    return 6;
  }

  const cudasim::ptx::Module device_module = cudasim::ptx::parse_ptx(R"PTX(
.version 8.0
.target sm_86
.address_size 64
.visible .entry entry_kernel(.param .u32 out)
{
	ret;
}
.visible .func (.param .b32 func_retval0) helper(.param .b32 x)
{
	ld.param.b32 %r1, [x];
	st.param.b32 [func_retval0+0], %r1;
	ret;
}
)PTX");
  if (device_module.functions.size() != 1) {
    return 7;
  }
  if (device_module.functions[0].name != "entry_kernel") {
    return 8;
  }

  return 0;
}
