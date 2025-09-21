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

        [[nodiscard]] void* Data() const { return m_Data.get(); }

        void* Data() { return m_Data.get(); }
    };
}