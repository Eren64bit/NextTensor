//
// Created by eren on 9/21/25.
//

#pragma once
#include "../core/NextTensor.h"

// ============== INTERNAL IMPLEMENTATION ==============
namespace Next::Ops::detail {
    template <typename T>
    void add_cpu_kernel_scalar(T* a, T* b, T* out, size_t size) {
        for (size_t i = 0; i < size; i++) {
            out[i] = a[i] + b[i];
        }
    }
}

namespace Next::Ops {
    template <typename T>
    NextTensor<T> Add(const NextTensor<T>& a, const NextTensor<T>& b) {
        if (a.Shape() != b.Shape()) {
            // TODO: add broadcasting
            throw std::runtime_error("Add: Shape mismatch");
        }

        // Result Tensor
        auto result = NextTensor<T>(a.Shape());

        Next::Ops::detail::add_cpu_kernel_scalar(
            static_cast<T*>(a.Data()),
            static_cast<T*>(b.Data()),
            static_cast<T*>(result.Data()),
            a.Size()
            );
        return result;
    }
}