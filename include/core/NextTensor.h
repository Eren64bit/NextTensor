//
// Created by eren on 9/21/25.
//

#pragma once

#include <memory>
#include "NextMetadata.h"

namespace Next {
    template<typename T>
    class NextTensor {
    private:
        NextMetadata m_Metadata;
        std::shared_ptr<T[]> m_Data;

        NextTensor(std::shared_ptr<T[]> data, NextMetadata metadata)
            : m_Metadata(std::move(metadata)), m_Data(std::move(data)) {}
    public:
        explicit NextTensor(const std::vector<size_t>& shape)
            : m_Metadata(shape, Next::TypeToDType<T>::value) {
            if (m_Metadata.Size() > 0) {
                m_Data = std::make_shared<T[]>(m_Metadata.Size());
            }
        }

        explicit NextTensor(const std::vector<size_t>& shape, const std::vector<size_t>& strides, size_t offset = 0)
            : m_Metadata(shape, strides, Next::TypeToDType<T>::value, offset) {
            if (m_Metadata.Size() > 0) {
                m_Data = std::make_shared<T[]>(m_Metadata.Size());
            }
        }

        [[nodiscard]] DType GetDType() const { return m_Metadata.GetDType(); }

        [[nodiscard]] const std::vector<size_t>& Shape() const { return m_Metadata.Shape(); }

        [[nodiscard]] const std::vector<size_t>& Strides() const { return m_Metadata.Strides(); }

        [[nodiscard]] size_t Offset() const { return m_Metadata.Offset(); }

        [[nodiscard]] size_t Size() const { return m_Metadata.Size(); }

        [[nodiscard]] size_t Rank() const { return m_Metadata.Rank(); }

        [[nodiscard]] bool IsContiguous() const { return  m_Metadata.IsContiguous(); }

        [[nodiscard]] T* Data() const { return m_Data.get(); }

        T* Data() { return m_Data.get(); }

        /**
         *  @brief Safe data access with bound check
         *
         * **/
        template<typename... Args>
        T& at(Args... args) {
            static_assert(sizeof...(args) > 0, "Attempt to access Tensor with wrong index count");

            if (m_Metadata.Rank() == 1 && sizeof...(args) == 1) {
                const auto idx = static_cast<size_t>(std::get<0>(std::tuple{args...}));
                if (idx >= m_Metadata.Shape()[0]) {
                    throw std::invalid_argument("Index out of range");
                }
                return m_Data.get()[m_Metadata.Offset() + idx];
            }

            if (sizeof...(args) != m_Metadata.Rank()) {
                throw std::invalid_argument("Error: 'at()' function" +
                                            std::to_string(m_Metadata.Rank()) + "expected but" +
                                            std::to_string(sizeof...(args)) + " got ");
            }

            static_assert((std::is_convertible_v<Args, size_t> && ...),
                      "All Indices must be convertible to size_t");

            std::array<size_t, sizeof...(Args)> indices = { static_cast<size_t>(args)... };

            size_t finalIndex = m_Metadata.Offset();
            const auto& shape = m_Metadata.Shape();
            const auto& strides = m_Metadata.Strides();

            for (size_t i = 0; i < indices.size(); i++) {
                if (indices[i] >= shape[i]) {
                    throw std::out_of_range("Index out of range");
                }

                finalIndex += indices[i] * strides[i];
            }

            return m_Data.get()[finalIndex];
        }

        /**
         *  @brief Safe data access with a bound check read-only version
         *
         * **/
        template<typename... Args>
        const T& at(Args... args) const {
            static_assert(sizeof...(args) > 0, "Attempt to access Tensor with wrong index count");

            if (m_Metadata.Rank() == 1 && sizeof...(args) == 1) {
                const auto idx = static_cast<size_t>(std::get<0>(std::tuple{args...}));
                if (idx >= m_Metadata.Shape()[0]) {
                    throw std::invalid_argument("Index out of range");
                }
                return m_Data.get()[m_Metadata.Offset() + idx];
            }

            if (sizeof...(args) != m_Metadata.Rank()) {
                throw std::invalid_argument("Error: 'at()' function" +
                                            std::to_string(m_Metadata.Rank()) + "expected but" +
                                            std::to_string(sizeof...(args)) + " got ");
            }

            static_assert((std::is_convertible_v<Args, size_t> && ...),
                      "All Indices must be convertible to size_t");

            std::array<size_t, sizeof...(Args)> indices = { static_cast<size_t>(args)... };

            size_t finalIndex = m_Metadata.Offset();
            const auto& shape = m_Metadata.Shape();
            const auto& strides = m_Metadata.Strides();

            for (size_t i = 0; i < indices.size(); i++) {
                if (indices[i] >= shape[i]) {
                    throw std::out_of_range("Index out of range");
                }

                finalIndex += indices[i] * strides[i];
            }

            return m_Data.get()[finalIndex];
        }

        /**
         *  @brief Fast access to data without bound check
         *
         * **/

        template<typename... Args>
        T& operator()(Args... args) {
            std::array<size_t, sizeof...(Args)> indices = { static_cast<size_t>(args)... };
            size_t finalIndex = m_Metadata.Offset();
            const auto& strides = m_Metadata.Strides();

            //TODO: add bound check for debug mode
            for (size_t i = 0; i < indices.size(); i++) {
                finalIndex += indices[i] * strides[i];
            }

            return m_Data.get()[finalIndex];
        }

        /**
         *  @brief Fast access to data without bound check read-only
         *
         * **/

        template<typename... Args>
        const T& operator()(Args... args) const {
            std::array<size_t, sizeof...(Args)> indices = { static_cast<size_t>(args)... };
            size_t finalIndex = m_Metadata.Offset();
            const auto& strides = m_Metadata.Strides();

            //TODO: add bound check for debug mode
            for (size_t i = 0; i < indices.size(); i++) {
                finalIndex += indices[i] * strides[i];
            }

            return m_Data.get()[finalIndex];
        }

        // TODO: Make these operator[] overload again when finished slice and view
        /**
         *  @brief Data access with scalar and bound check
         *
         * **/
        T& operator[](size_t idx) {
            if (!IsContiguous()) {
                throw std::invalid_argument("Tensor is not contiguous");
            }
            if (idx >= m_Metadata.Size()) {
                throw std::invalid_argument("Index out of range");
            }
            return m_Data.get()[idx];
        }
        /**
         *  @brief Data access with scalar and bound check read-only version
         *
         * **/
        const T& operator[](size_t idx) const {
            if (!IsContiguous()) {
                throw std::invalid_argument("Tensor is not contiguous");
            }
            if (idx >= m_Metadata.Size()) {
                throw std::invalid_argument("Index out of range");
            }
            return m_Data.get()[idx];
        }

        /**
         *  @brief fill function
         * **/

        void fill(const T& value) {
            if (Size() == 0) return;
            T* dataPtr = this->Data();

            if (IsContiguous()) {
                std::fill(dataPtr + Offset(), dataPtr + Offset() + Size(), value);
            }
            else {
                std::vector<size_t> indices(Rank(), 0);
                const auto& shape = m_Metadata.Shape();
                const auto& strides = m_Metadata.Strides();

                for (size_t i = 0; i < Size(); i++) {
                    size_t finalIndex = Offset();
                    for (size_t d = 0; d < Rank(); ++d) {
                        finalIndex += indices[d] * strides[d];
                    }
                    dataPtr[finalIndex] = value;

                    for (int d = Rank() - 1; d >= 0; --d) {
                        if (++indices[d] < shape[d]) {
                            break;
                        }
                        indices[d] = 0;
                    }
                }
            }
        }

        void zeros() {
            fill(T{});
        }

        void ones() {
            static_assert(std::is_constructible_v<T, int>, "T must be constructible from '1' (int) to use Ones()");
            fill(T{1});
        }

        //VIEW operations
        NextTensor<T> reshape(const std::vector<size_t>& shape) {
            if (!IsContiguous()) {
                throw std::runtime_error("Tensor is not contiguous {Reshape}");
            }
            if (Size() != Next::ComputeSize(shape)) {
                throw std::runtime_error("Reshape shape is not compatible with current tensor shape");
            }
            NextMetadata n_Metadata{shape, Next::ComputeStrides(shape), this->GetDType(), this->Offset()};

            return NextTensor<T>{this->m_Data, n_Metadata};
        }

        NextTensor<T> transpose(const size_t& dim1, const size_t& dim2) {
            if (dim1 >= Rank() || dim2 >= Rank()) {
                throw std::runtime_error("Transpose dimensions are out of range");
            }
            auto n_Shape = Shape();
            auto n_Strides = Strides();
            std::swap(n_Shape[dim1], n_Shape[dim2]);
            std::swap(n_Strides[dim1], n_Strides[dim2]);
            NextMetadata n_Metadata{n_Shape, n_Strides, this->GetDType(), this->Offset()};

            return NextTensor<T>{this->m_Data, n_Metadata};
        }

        NextTensor<T> slice(const size_t& dim = 0, const size_t& start = 0, const size_t& end = 0) {
            if (dim >= Rank()) {
                throw std::out_of_range("Slice error: Dimension index (" + std::to_string(dim) +
                                         ") is out of range for tensor with rank " + std::to_string(Rank()));
            }

            if (start > end) {
                throw std::runtime_error("Slice error: 'start' index (" + std::to_string(start) +
                                         ") cannot be greater than 'end' index (" + std::to_string(end) + ").");
            }

            if (end > Shape()[dim]) {
                throw std::out_of_range("Slice error: 'end' index (" + std::to_string(end) +
                                         ") is out of bounds for dimension " + std::to_string(dim) +
                                         " (size is " + std::to_string(Shape()[dim]) + ").");
            }

            auto n_Shape = Shape();
            n_Shape[dim] = end - start;

            auto n_Offset = Offset() + start * Strides()[dim];

            NextMetadata n_Metadata{n_Shape, Strides(), this->GetDType(), n_Offset};
            return NextTensor<T>{this->m_Data, n_Metadata};
        }

        //Tensor element-wise operations
        NextTensor<T>& operator+=(const NextTensor<T>& other) {
            if (this->Shape() != other.Shape()) {
                throw std::runtime_error("Tensor shapes are not compatible for InPlace addition");
            }
            if (this->IsContiguous() && other.IsContiguous()) {
                for (size_t i = 0; i < Size(); i++) {
                    this->Data()[i + this->Offset()] += other.Data()[i + other.Offset()];
                }
            } else {
                std::vector<size_t> indices(Rank(), 0);
                const auto& shape = this->Shape();
                for (size_t i = 0; i < Size(); i++) {
                    size_t indexA = this->Offset() + Next::FlattenIndex(this->Strides(), indices);
                    size_t indexB = other.Offset() + Next::FlattenIndex(other.Strides(), indices);
                    this->Data()[indexA] += other.Data()[indexB];
                    for (int d = Rank() - 1; d >= 0; --d) {
                        if (++indices[d] < shape[d]) {
                            break;
                        }
                        indices[d] = 0;
                    }
                }
            }
            return *this;
        }

        NextTensor<T>& operator+=(const T& other) {
            if (this->IsContiguous()) {
                for (size_t i = 0; i < Size(); i++) {
                    this->Data()[i + this->Offset()] += other;
                }
            } else {
                std::vector<size_t> indices(Rank(), 0);
                for (size_t i = 0; i < Size(); i++) {
                    size_t indexA = this->Offset() + Next::FlattenIndex(this->Strides(), indices);
                    this->Data()[indexA] += other;
                    for (int d = Rank() - 1; d >= 0; --d) {
                        if (++indices[d] < Shape()[d]) {
                            break;
                        }
                        indices[d] = 0;
                    }
                }
            }
            return *this;
        }

        NextTensor<T>& operator-=(const NextTensor<T>& other) {
            if (this->Shape() != other.Shape()) {
                throw std::runtime_error("Tensor shapes are not compatible for InPlace addition");
            }
            if (this->IsContiguous() && other.IsContiguous()) {
                for (size_t i = 0; i < Size(); i++) {
                    this->Data()[i + this->Offset()] -= other.Data()[i + other.Offset()];
                }
            } else {
                std::vector<size_t> indices(Rank(), 0);
                const auto& shape = this->Shape();
                for (size_t i = 0; i < Size(); i++) {
                    size_t indexA = this->Offset() + Next::FlattenIndex(this->Strides(), indices);
                    size_t indexB = other.Offset() + Next::FlattenIndex(other.Strides(), indices);
                    this->Data()[indexA] -= other.Data()[indexB];
                    for (int d = Rank() - 1; d >= 0; --d) {
                        if (++indices[d] < shape[d]) {
                            break;
                        }
                        indices[d] = 0;
                    }
                }
            }
            return *this;
        }

        NextTensor<T>& operator-=(const T& other) {
            if (this->IsContiguous()) {
                for (size_t i = 0; i < Size(); i++) {
                    this->Data()[i + this->Offset()] -= other;
                }
            } else {
                std::vector<size_t> indices(Rank(), 0);
                for (size_t i = 0; i < Size(); i++) {
                    size_t indexA = this->Offset() + Next::FlattenIndex(this->Strides(), indices);
                    this->Data()[indexA] -= other;
                    for (int d = Rank() - 1; d >= 0; --d) {
                        if (++indices[d] < Shape()[d]) {
                            break;
                        }
                        indices[d] = 0;
                    }
                }
            }
            return *this;
        }

        NextTensor<T>& operator*=(const NextTensor<T>& other) {
            if (this->Shape() != other.Shape()) {
                throw std::runtime_error("Tensor shapes are not compatible for InPlace addition");
            }
            if (this->IsContiguous() && other.IsContiguous()) {
                for (size_t i = 0; i < Size(); i++) {
                    this->Data()[i + this->Offset()] *= other.Data()[i + other.Offset()];
                }
            } else {
                std::vector<size_t> indices(Rank(), 0);
                const auto& shape = this->Shape();
                for (size_t i = 0; i < Size(); i++) {
                    size_t indexA = this->Offset() + Next::FlattenIndex(this->Strides(), indices);
                    size_t indexB = other.Offset() + Next::FlattenIndex(other.Strides(), indices);
                    this->Data()[indexA] *= other.Data()[indexB];
                    for (int d = Rank() - 1; d >= 0; --d) {
                        if (++indices[d] < shape[d]) {
                            break;
                        }
                        indices[d] = 0;
                    }
                }
            }
            return *this;
        }

        NextTensor<T>& operator*=(const T& other) {
            if (this->IsContiguous()) {
                for (size_t i = 0; i < Size(); i++) {
                    this->Data()[i + this->Offset()] *= other;
                }
            } else {
                std::vector<size_t> indices(Rank(), 0);
                for (size_t i = 0; i < Size(); i++) {
                    size_t indexA = this->Offset() + Next::FlattenIndex(this->Strides(), indices);
                    this->Data()[indexA] *= other;
                    for (int d = Rank() - 1; d >= 0; --d) {
                        if (++indices[d] < Shape()[d]) {
                            break;
                        }
                        indices[d] = 0;
                    }
                }
            }
            return *this;
        }

        NextTensor<T>& operator/=(const NextTensor<T>& other) {
            if (this->Shape() != other.Shape()) {
                throw std::runtime_error("Tensor shapes are not compatible for InPlace addition");
            }
            if (this->IsContiguous() && other.IsContiguous()) {
                for (size_t i = 0; i < Size(); i++) {
                    if (other.Data()[i + other.Offset()] == 0) {
                        throw std::runtime_error("Division by zero");
                    }
                    this->Data()[i + this->Offset()] /= other.Data()[i + other.Offset()];
                }
            } else {
                std::vector<size_t> indices(Rank(), 0);
                const auto& shape = this->Shape();
                for (size_t i = 0; i < Size(); i++) {
                    size_t indexA = this->Offset() + Next::FlattenIndex(this->Strides(), indices);
                    size_t indexB = other.Offset() + Next::FlattenIndex(other.Strides(), indices);
                    if (other.Data()[indexB] == 0) {
                        throw std::runtime_error("Division by zero");
                    }
                    this->Data()[indexA] /= other.Data()[indexB];
                    for (int d = Rank() - 1; d >= 0; --d) {
                        if (++indices[d] < shape[d]) {
                            break;
                        }
                        indices[d] = 0;
                    }
                }
            }
            return *this;
        }

        NextTensor<T>& operator/=(const T& other) {
            if (this->IsContiguous()) {
                for (size_t i = 0; i < Size(); i++) {
                    if (other == 0) {
                        throw std::runtime_error("Division by zero");
                    }
                    this->Data()[i + this->Offset()] /= other;
                }
            } else {
                std::vector<size_t> indices(Rank(), 0);
                for (size_t i = 0; i < Size(); i++) {
                    if (other == 0) {
                        throw std::runtime_error("Division by zero");
                    }
                    size_t indexA = this->Offset() + Next::FlattenIndex(this->Strides(), indices);
                    this->Data()[indexA] /= other;
                    for (int d = Rank() - 1; d >= 0; --d) {
                        if (++indices[d] < Shape()[d]) {
                            break;
                        }
                        indices[d] = 0;
                    }
                }
            }
            return *this;
        }

        // Element-wise Operations
        // Tensor plus Tensor
        NextTensor<T> add(const NextTensor<T>& other) const {
            if (this->Shape() != other.Shape()) {
                //TODO: Add broadcasting
                throw std::runtime_error("Tensor shapes are not compatible for element-wise addition");
            }
            NextTensor<T> resultTensor{this->Shape()}; // Result Tensor
            if (this->IsContiguous() && other.IsContiguous()) {
                for (size_t i = 0; i < Size(); i++) {
                    resultTensor.Data()[i] = this->Data()[i + this->Offset()] + other.Data()[i + other.Offset()];
                }
            } else {
                std::vector<size_t> indices(Rank(), 0);
                const auto& shape = this->Shape();
                for (size_t i = 0; i < Size(); i++) {
                    size_t indexA = this->Offset() + Next::FlattenIndex(this->Strides(), indices);
                    size_t indexB = other.Offset() + Next::FlattenIndex(other.Strides(), indices);
                    resultTensor.Data()[i] = this->Data()[indexA] + other.Data()[indexB];
                    for (int d = Rank() - 1; d >= 0; --d) {
                        if (++indices[d] < shape[d]) {
                            break;
                        }
                        indices[d] = 0;
                    }
                }
            }
            return resultTensor;
        }

        NextTensor<T> add(const T& scalar) const {
            NextTensor<T> resultTensor{this->Shape()};
            if (this->IsContiguous()) {
                for (size_t i = 0; i < Size(); i++) {
                    resultTensor.Data()[i] = this->Data()[i + this->Offset()] + scalar;
                }
            } else {
                std::vector<size_t> indices(Rank(), 0);
                for (size_t i = 0; i < Size(); i++) {
                    size_t indexA = this->Offset() + Next::FlattenIndex(this->Strides(), indices);
                    resultTensor.Data()[i] = this->Data()[indexA] + scalar;
                    for (int d = Rank() - 1; d >= 0; --d) {
                        if (++indices[d] < Shape()[d]) {
                            break;
                        }
                        indices[d] = 0;
                    }
                }
            }
            return resultTensor;
        }

        NextTensor<T> sub(const NextTensor<T>& other) const {
            if (this->Shape() != other.Shape()) {
                //TODO: Add broadcasting
                throw std::runtime_error("Tensor shapes are not compatible for element-wise addition");
            }
            NextTensor<T> resultTensor{this->Shape()}; // Result Tensor
            if (this->IsContiguous() && other.IsContiguous()) {
                for (size_t i = 0; i < Size(); i++) {
                    resultTensor.Data()[i] = this->Data()[i + this->Offset()] - other.Data()[i + other.Offset()];
                }
            } else {
                std::vector<size_t> indices(Rank(), 0);
                const auto& shape = this->Shape();
                for (size_t i = 0; i < Size(); i++) {
                    size_t indexA = this->Offset() + Next::FlattenIndex(this->Strides(), indices);
                    size_t indexB = other.Offset() + Next::FlattenIndex(other.Strides(), indices);
                    resultTensor.Data()[i] = this->Data()[indexA] - other.Data()[indexB];
                    for (int d = Rank() - 1; d >= 0; --d) {
                        if (++indices[d] < shape[d]) {
                            break;
                        }
                        indices[d] = 0;
                    }
                }
            }
            return resultTensor;
        }

        NextTensor<T> sub(const T& scalar) const {
            NextTensor<T> resultTensor{this->Shape()};
            if (this->IsContiguous()) {
                for (size_t i = 0; i < Size(); i++) {
                    resultTensor.Data()[i] = this->Data()[i + this->Offset()] - scalar;
                }
            } else {
                std::vector<size_t> indices(Rank(), 0);
                for (size_t i = 0; i < Size(); i++) {
                    size_t indexA = this->Offset() + Next::FlattenIndex(this->Strides(), indices);
                    resultTensor.Data()[i] = this->Data()[indexA] - scalar;
                    for (int d = Rank() - 1; d >= 0; --d) {
                        if (++indices[d] < Shape()[d]) {
                            break;
                        }
                        indices[d] = 0;
                    }
                }
            }
            return resultTensor;
        }

        NextTensor<T> rsub(const T& scalar) const {
            NextTensor<T> resultTensor{this->Shape()};
            if (this->IsContiguous()) {
                for (size_t i = 0; i < Size(); i++) {
                    resultTensor.Data()[i] = scalar - this->Data()[i + this->Offset()];
                }
            } else {
                std::vector<size_t> indices(Rank(), 0);
                for (size_t i = 0; i < Size(); i++) {
                    size_t indexA = this->Offset() + Next::FlattenIndex(this->Strides(), indices);
                    resultTensor.Data()[i] = scalar - this->Data()[indexA];
                    for (int d = Rank() - 1; d >= 0; --d) {
                        if (++indices[d] < Shape()[d]) {
                            break;
                        }
                        indices[d] = 0;
                    }
                }
            }
            return resultTensor;
        }

        NextTensor<T> mult(const NextTensor<T>& other) const {
            if (this->Shape() != other.Shape()) {
                //TODO: Add broadcasting
                throw std::runtime_error("Tensor shapes are not compatible for element-wise addition");
            }
            NextTensor<T> resultTensor{this->Shape()}; // Result Tensor
            if (this->IsContiguous() && other.IsContiguous()) {
                for (size_t i = 0; i < Size(); i++) {
                    resultTensor.Data()[i] = this->Data()[i + this->Offset()] * other.Data()[i + other.Offset()];
                }
            } else {
                std::vector<size_t> indices(Rank(), 0);
                const auto& shape = this->Shape();
                for (size_t i = 0; i < Size(); i++) {
                    size_t indexA = this->Offset() + Next::FlattenIndex(this->Strides(), indices);
                    size_t indexB = other.Offset() + Next::FlattenIndex(other.Strides(), indices);
                    resultTensor.Data()[i] = this->Data()[indexA] * other.Data()[indexB];
                    for (int d = Rank() - 1; d >= 0; --d) {
                        if (++indices[d] < shape[d]) {
                            break;
                        }
                        indices[d] = 0;
                    }
                }
            }
            return resultTensor;
        }

        NextTensor<T> mult(const T& scalar) const {
            NextTensor<T> resultTensor{this->Shape()};
            if (this->IsContiguous()) {
                for (size_t i = 0; i < Size(); i++) {
                    resultTensor.Data()[i] = this->Data()[i + this->Offset()] * scalar;
                }
            } else {
                std::vector<size_t> indices(Rank(), 0);
                for (size_t i = 0; i < Size(); i++) {
                    size_t indexA = this->Offset() + Next::FlattenIndex(this->Strides(), indices);
                    resultTensor.Data()[i] = this->Data()[indexA] * scalar;
                    for (int d = Rank() - 1; d >= 0; --d) {
                        if (++indices[d] < Shape()[d]) {
                            break;
                        }
                        indices[d] = 0;
                    }
                }
            }
            return resultTensor;
        }

        NextTensor<T> divide(const NextTensor<T>& other) const {
            if (this->Shape() != other.Shape()) {
                //TODO: Add broadcasting
                throw std::runtime_error("Tensor shapes are not compatible for element-wise addition");
            }
            NextTensor<T> resultTensor{this->Shape()}; // Result Tensor
            if (this->IsContiguous() && other.IsContiguous()) {
                for (size_t i = 0; i < Size(); i++) {
                    if (other.Data()[i + other.Offset()] == 0) {
                        throw std::runtime_error("Division by zero");
                    }
                    resultTensor.Data()[i] = this->Data()[i + this->Offset()] / other.Data()[i + other.Offset()];
                }
            } else {
                std::vector<size_t> indices(Rank(), 0);
                const auto& shape = this->Shape();
                for (size_t i = 0; i < Size(); i++) {
                    size_t indexA = this->Offset() + Next::FlattenIndex(this->Strides(), indices);
                    size_t indexB = other.Offset() + Next::FlattenIndex(other.Strides(), indices);
                    if (other.Data()[indexB] == 0) {
                        throw std::runtime_error("Division by zero");
                    }
                    resultTensor.Data()[i] = this->Data()[indexA] / other.Data()[indexB];
                    for (int d = Rank() - 1; d >= 0; --d) {
                        if (++indices[d] < shape[d]) {
                            break;
                        }
                        indices[d] = 0;
                    }
                }
            }
            return resultTensor;
        }

        NextTensor<T> divide(const T& scalar) const {
            NextTensor<T> resultTensor{this->Shape()};
            if (this->IsContiguous()) {
                for (size_t i = 0; i < Size(); i++) {
                    if (scalar == 0) {
                        throw std::runtime_error("Division by zero");
                    }
                    resultTensor.Data()[i] = this->Data()[i + this->Offset()] / scalar;
                }
            } else {
                std::vector<size_t> indices(Rank(), 0);
                for (size_t i = 0; i < Size(); i++) {
                    if (scalar == 0) {
                        throw std::runtime_error("Division by zero");
                    }
                    size_t indexA = this->Offset() + Next::FlattenIndex(this->Strides(), indices);
                    resultTensor.Data()[i] = this->Data()[indexA] / scalar;
                    for (int d = Rank() - 1; d >= 0; --d) {
                        if (++indices[d] < Shape()[d]) {
                            break;
                        }
                        indices[d] = 0;
                    }
                }
            }
            return resultTensor;
        }

        NextTensor<T> rdivide(const T& scalar) const {
            NextTensor<T> resultTensor{this->Shape()};
            if (this->IsContiguous()) {
                for (size_t i = 0; i < Size(); i++) {
                    if (this->Data()[i + this->Offset()] == 0) {
                        throw std::runtime_error("Division by zero");
                    }
                    resultTensor.Data()[i] = scalar / this->Data()[i + this->Offset()];
                }
            } else {
                std::vector<size_t> indices(Rank(), 0);
                for (size_t i = 0; i < Size(); i++) {
                    size_t indexA = this->Offset() + Next::FlattenIndex(this->Strides(), indices);
                    if (this->Data()[indexA] == 0) {
                        throw std::runtime_error("Division by zero");
                    }
                    resultTensor.Data()[i] = scalar / this->Data()[indexA];
                    for (int d = Rank() - 1; d >= 0; --d) {
                        if (++indices[d] < Shape()[d]) {
                            break;
                        }
                        indices[d] = 0;
                    }
                }
            }
            return resultTensor;
        }
    };
}