#include <catch2/catch_test_macros.hpp>

#include "golden/golden.hpp"

TEST_CASE("format du manifest") { REQUIRE(golden::manifest_format_version() == 1); }
