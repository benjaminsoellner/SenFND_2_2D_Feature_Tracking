cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(camera_fusion)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")


find_package(OpenCV 4.1 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})


# Quizzes Lesson 2

# Executable for create matrix exercise
add_executable (q21_create_matrix src/quizzes/q21_create_matrix.cpp)
target_link_libraries (q21_create_matrix ${OpenCV_LIBRARIES})

# Executable for change pixels exercise
add_executable (q22_change_pixels src/quizzes/q22_change_pixels.cpp)
target_link_libraries (q22_change_pixels ${OpenCV_LIBRARIES})

# Executables for load image exercises
add_executable (q23_load_image_1 src/quizzes/q23_load_image_1.cpp)
target_link_libraries(q23_load_image_1 ${OpenCV_LIBRARIES})

add_executable (q24_load_image_2 src/quizzes/q24_load_image_2.cpp)
target_link_libraries(q24_load_image_2 ${OpenCV_LIBRARIES})

add_executable (q25_load_image_3 src/quizzes/q25_load_image_3.cpp)
target_link_libraries(q25_load_image_3 ${OpenCV_LIBRARIES})

# Quizzes Lesson 3

# Executables for TTC with lidar exercise
add_executable (q31_compute_ttc_lidar src/quizzes/q31_compute_ttc_lidar.cpp src/quizzes/structIO.cpp)
target_link_libraries(q31_compute_ttc_lidar ${OpenCV_LIBRARIES})

# Executables for TTC with camera exercise
add_executable (q32_compute_ttc_camera src/quizzes/q32_compute_ttc_camera.cpp src/quizzes/structIO.cpp)
target_link_libraries(q32_compute_ttc_camera ${OpenCV_LIBRARIES})

# Quizzes Lesson 4

# Executables for Gaussian Smoothing Quiz
add_executable (q41_gaussian_smoothing src/quizzes/q41_gaussian_smoothing.cpp)
target_link_libraries(q41_gaussian_smoothing ${OpenCV_LIBRARIES})

# Executables for Magnitude Sobel Operator
add_executable (q42_magnitude_sobel src/quizzes/q42_magnitude_sobel.cpp)
target_link_libraries(q42_magnitude_sobel ${OpenCV_LIBRARIES})

# Executables for Harris Detector
add_executable (q43_cornerness_harris src/quizzes/q43_cornerness_harris.cpp)
target_link_libraries(q43_cornerness_harris ${OpenCV_LIBRARIES})

# Executables for Shi-Tomasi & FAST Detector
add_executable (q44_detect_keypoints src/quizzes/q44_detect_keypoints.cpp)
target_link_libraries(q44_detect_keypoints ${OpenCV_LIBRARIES})

# Executables for SIFT & BRISK Detector
add_executable (q45_describe_keypoints src/quizzes/q45_describe_keypoints.cpp)
target_link_libraries(q45_describe_keypoints ${OpenCV_LIBRARIES})

# Executables for Descriptor Matching
add_executable (q46_descriptor_matching src/quizzes/q46_descriptor_matching.cpp src/quizzes/structIO.cpp)
target_link_libraries(q46_descriptor_matching ${OpenCV_LIBRARIES})

# Mid-Term Project

# Executable for project
add_executable (2D_feature_tracking src/matching2D_Student.cpp src/MidTermProject_Camera_Student.cpp)
target_link_libraries (2D_feature_tracking ${OpenCV_LIBRARIES})

# IDE support for VSCode
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )