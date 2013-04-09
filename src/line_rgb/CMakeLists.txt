find_package(OpenCV REQUIRED)

add_library(opencv_line_rgb src/line_rgb.cpp
)

target_link_libraries(opencv_line_rgb ${OpenCV_LIBS}
                                      ${OpenCV_LIBRARIES}
)

install(TARGETS opencv_line_rgb
        DESTINATION lib
)

install(DIRECTORY include/opencv2/opencv_line_rgb
        DESTINATION include/opencv2
        COMPONENT main
)

# Add sample
add_executable(line_rgb_sample samples/line_test.cpp)
target_link_libraries(line_rgb_sample ${OpenCV_LIBRARIES} opencv_line_rgb)


