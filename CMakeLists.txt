cmake_minimum_required(VERSION 3.10)

project(FirstProject2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(OpenCV_DIR C:/opencv/build)

find_package(OpenCV REQUIRED)

# Added executable for imgDisplay.cp

add_executable(computeFeatures src/computeFeatures.cpp src/kmeans.cpp src/objectRecogFunctions.cpp)
target_link_libraries(computeFeatures ${OpenCV_LIBS})

add_executable(classifyImages src/classifyImages.cpp src/kmeans.cpp src/objectRecogFunctions.cpp)
target_link_libraries(classifyImages ${OpenCV_LIBS})

add_executable(classifyImagesKNN src/classifyImagesKNN.cpp src/kmeans.cpp src/objectRecogFunctions.cpp)
target_link_libraries(classifyImagesKNN ${OpenCV_LIBS})

add_executable(classifyImagesDNN src/classifyImagesDNN.cpp src/kmeans.cpp src/objectRecogFunctions.cpp)
target_link_libraries(classifyImagesDNN ${OpenCV_LIBS})

add_executable(segmentImages src/segmentImages.cpp src/kmeans.cpp src/objectRecogFunctions.cpp)
target_link_libraries(segmentImages ${OpenCV_LIBS})

add_executable(segmentImagesCV src/segmentImagesCV.cpp src/kmeans.cpp src/objectRecogFunctions.cpp)
target_link_libraries(segmentImagesCV ${OpenCV_LIBS})

add_executable(morphologicalFilter src/morphologicalFilter.cpp src/kmeans.cpp src/objectRecogFunctions.cpp)
target_link_libraries(morphologicalFilter ${OpenCV_LIBS})

add_executable(threshold src/threshold.cpp src/kmeans.cpp src/objectRecogFunctions.cpp)
target_link_libraries(threshold ${OpenCV_LIBS})

add_executable(trainingData src/trainingData.cpp src/kmeans.cpp src/objectRecogFunctions.cpp)
target_link_libraries(trainingData ${OpenCV_LIBS})

add_executable(trainingDataDNN src/trainingDataDNN.cpp src/kmeans.cpp src/objectRecogFunctions.cpp)
target_link_libraries(trainingDataDNN ${OpenCV_LIBS})

add_executable(ImgDisplay src/imgDisplay.cpp)
target_link_libraries(ImgDisplay ${OpenCV_LIBS})

