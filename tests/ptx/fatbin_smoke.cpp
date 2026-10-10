#include "loader/fatbin.h"

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
  const std::string ptx = buffer.str();

  cudasim::loader::FatbinResult result =
      cudasim::loader::extract_ptx(ptx.data(), ptx.size());
  if (result.cubin_only) {
    return 2;
  }
  if (result.ptx_sources.size() != 1) {
    return 3;
  }

  const char binary[] = {0x01, 0x02, 0x03, 0x04};
  cudasim::loader::FatbinResult cubin =
      cudasim::loader::extract_ptx(binary, sizeof(binary));
  if (!cubin.cubin_only) {
    return 4;
  }

  return 0;
}
