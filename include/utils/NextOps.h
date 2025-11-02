//
// Created by eren on 11/1/25.
//

#pragma once
#include "../core/NextTensor.h"

namespace Next {
    template<typename T>
    NextTensor<T> operator+(const NextTensor<T>& lhs, const NextTensor<T>& rhs) {
        return lhs.add(rhs);
    }
    template<typename T>
    NextTensor<T> operator+(const NextTensor<T>& lhs, const T& rhs) {
        return lhs.add(rhs);
    }

    template<typename T>
    NextTensor<T> operator+(const T& scalar, const NextTensor<T>& rhs) {
        return rhs.add(scalar);
    }

    template<typename T>
    NextTensor<T> operator-(const NextTensor<T>& lhs, const NextTensor<T>& rhs) {
        return lhs.sub(rhs);
    }

    template<typename T>
    NextTensor<T> operator-(const NextTensor<T>& lhs, const T& rhs) {
        return lhs.sub(rhs);
    }

    template<typename T>
    NextTensor<T> operator-(const T& scalar, const NextTensor<T>& rhs) {
        return rhs.rsub(scalar);
    }

    template<typename T>
    NextTensor<T> operator*(const NextTensor<T>& lhs, const NextTensor<T>& rhs) {
        return lhs.mult(rhs);
    }

    template<typename T>
    NextTensor<T> operator*(const NextTensor<T>& lhs, const T& rhs) {
        return lhs.mult(rhs);
    }

    template<typename T>
    NextTensor<T> operator*(const T& scalar, const NextTensor<T>& rhs) {
        return rhs.mult(scalar);
    }

    template<typename T>
    NextTensor<T> operator/(const NextTensor<T>& lhs, const NextTensor<T>& rhs) {
        return lhs.divide(rhs);
    }

    template<typename T>
    NextTensor<T> operator/(const NextTensor<T>& lhs, const T& rhs) {
        return lhs.divide(rhs);
    }

    template<typename T>
    NextTensor<T> operator/(const T& scalar, const NextTensor<T>& rhs) {
        return rhs.rdivide(scalar);
    }
}