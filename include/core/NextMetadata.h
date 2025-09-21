//
// Created by eren on 9/20/25.
//
#pragma once
#include <vector>

#include "../utils/NextUtils.h"

namespace Next {
    //*
    //@brief A class to hold metadata about a tensor, including its shape, strides, offset, total size, and contiguity.
    //*/
    class NextMetadata {
    protected:
        DType m_DType;                      // Type of the tensor data (e.g., float32, float64, int32)
        std::vector<size_t> m_Shape;        // Shape of the tensor (e.g., {2, 3, 4} for a 2x3x4 tensor)
        std::vector<size_t> m_Strides;      // Strides of the tensor (e.g., {12, 4, 1} for a 2x3x4 tensor)
        size_t m_Offset{0};                 // Offset in the underlying data array
        size_t m_Size{0};                   // Total number of elements in the tensor
        size_t m_Rank{0};                   // Rank (number of dimensions) of the tensor
        bool m_Contiguous{true};            // Whether the tensor is stored in contiguous memory
    public:
        explicit NextMetadata(const std::vector<size_t>& shape, DType dtype, size_t offset = 0)
            : m_Shape(shape), m_DType(dtype), m_Offset(offset) {
            m_Strides = Next::ComputeStrides(m_Shape);
            m_Size = Next::ComputeSize(m_Shape);
            m_Rank = m_Shape.size();
            m_Contiguous = Next::IsContiguous(m_Shape, m_Strides);
        }

        NextMetadata(const std::vector<size_t>& shape, const std::vector<size_t>& strides, DType dtype, size_t offset = 0)
            : m_Shape(shape), m_Strides(strides), m_DType(dtype), m_Offset(offset) {
            m_Size = Next::ComputeSize(m_Shape);
            m_Rank = m_Shape.size();
            m_Contiguous = Next::IsContiguous(m_Shape, m_Strides);
        }

        [[nodiscard]] DType GetDType() const { return m_DType; }

        [[nodiscard]] const std::vector<size_t>& Shape() const { return m_Shape; }

        [[nodiscard]] const std::vector<size_t>& Strides() const { return m_Strides; }

        [[nodiscard]] size_t Offset() const { return m_Offset; }

        [[nodiscard]] size_t Size() const { return m_Size; }

        [[nodiscard]] size_t Rank() const { return m_Rank; }

        [[nodiscard]] bool IsContiguous() const { return m_Contiguous; }
    };
}
