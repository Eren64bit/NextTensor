//
// Created by eren on 9/20/25.
//

#pragma once
#include <vector>

namespace Next {
    [[nodiscard]] inline  std::vector<size_t> ComputeStrides(const std::vector<size_t>& strides) {
        std::vector<size_t> result(strides.size(), 1);
        for (size_t i = 0; i < strides.size(); i++) {
            result[i] = result[i] * strides[i];
        }
        return result;
    }

    [[nodiscard]] inline size_t ComputeSize(const std::vector<size_t>& shape) {
        size_t total = 1;
        for (auto& s : shape) {
            total *= s;
        }
        return total;
    }

    [[nodiscard]] inline size_t FlattenIndex(const std::vector<size_t>& strides, std::vector<size_t>& indices) {
        size_t FlattenIndex = 0;
        for (size_t i = 0; i < stride.size(); i++) {
            FlattenIndex += stride[i] * indices[i];
        }
        return FlattenIndex;
    }

    [[nodiscard]] inline std::vector<size_t> UnflattenIndex(const std::vector<size_t>& strides, size_t index) {
        std::vector<size_t> result(strides.size());
        for (size_t i = 0; i < strides.size(); i++) {
            result[i] = index / strides[i];
            index %= strides[i];
        }
        return result;
    }
}
