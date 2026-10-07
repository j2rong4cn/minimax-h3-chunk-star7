#pragma once
#include <sstream>
#include <stdexcept>

template <typename... Args>
inline void veda_cuda_check(bool ok, const Args&... args) {
    if (ok) return;
    std::ostringstream message;
    (message << ... << args);
    throw std::runtime_error(message.str());
}
#define VEDA_CHECK(...) veda_cuda_check(__VA_ARGS__)
