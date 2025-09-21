//
// Created by eren on 9/20/25.
//

#pragma once
#include <vector>
#include <stdexcept>
#include "DType.h"

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
        for (size_t i = 0; i < strides.size(); i++) {
            FlattenIndex += strides[i] * indices[i];
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

    [[nodiscard]] inline bool IsContiguous(const std::vector<size_t>& shape, const std::vector<size_t>& strides) {

        if (shape.size() != strides.size()) throw std::invalid_argument("Shape and Strides Ranks must be match");

        if (strides.back() != 1) throw std::invalid_argument("Last element of stride must be one.");

        if (shape.empty()) return true;

        for (int i = (int)strides.size() - 1; i >= 1; i--) {
            if (strides[i-1] != strides[i] * shape[i]) return false;
        }
        return true;
    }

    template <typename T>
    struct TypeToDType {
        static constexpr DType value = DType::UNKNOWN;
    };

    template <>
    struct TypeToDType<float> {
        static constexpr DType value = DType::FLOAT32;
    };

    template <>
    struct  TypeToDType<double> {
        static constexpr DType value = DType::FLOAT64;
    };

    template <>
    struct  TypeToDType<int32_t> {
        static constexpr DType value = DType::INT32;
    };

    template <>
    struct TypeToDType<int64_t> {
        static constexpr DType value = DType::INT64;
    };

    template<>
    struct TypeToDType<uint8_t> {
        static constexpr DType value = DType::UINT8;
    };

    template<>
    struct TypeToDType<bool> {
        static constexpr DType value = DType::BOOL;
    };
}
