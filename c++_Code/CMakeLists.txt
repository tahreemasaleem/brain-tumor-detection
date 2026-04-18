cmake_minimum_required(VERSION 3.18)
project(TumorClassification)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(tumor_classifier
    main.cpp
    dataset.cpp
    vgg16.cpp
    trainer.cpp
)

target_link_libraries(tumor_classifier 
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)