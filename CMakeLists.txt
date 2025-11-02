cmake_minimum_required(VERSION 4.0)
project(NextTensor)

set(CMAKE_CXX_STANDARD 20)

add_library(NextTensor STATIC
        include/core/NextMetadata.h
        src/test/library.cpp
        include/utils/NextUtils.h
        include/core/NextTensor.h
        include/utils/DType.h
        include/utils/NextOps.h
        include/utils/BroadcastUtils.h
)
