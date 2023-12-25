cmake_minimum_required(VERSION 3.10)
set(PROJECT_NAME clipTokenizer)  # 设置项目名称变量

project(${PROJECT_NAME})

file(COPY ${CMAKE_SOURCE_DIR}/vocab.txt DESTINATION ${CMAKE_BINARY_DIR})

find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
    message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
else ()
    message(FATAL_ERROR "opencv Not Found!")
endif (OpenCV_FOUND)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    message(FATAL_ERROR "OpenMP Not Found!")
endif ()

#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/ncnn)
#link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

FIND_PACKAGE(ncnn REQUIRED)

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
)


# 添加ICU库
find_package(ICU REQUIRED COMPONENTS uc i18n data)

add_executable(${PROJECT_NAME} ${SOURCES})

# 链接ICU库
target_link_libraries(${PROJECT_NAME} PRIVATE ICU::uc ICU::i18n ICU::data ncnn ${OpenCV_LIBS})


