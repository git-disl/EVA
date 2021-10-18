# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME object_detection_demo_yolov3_async
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/object_detection_demo_yolov3_async.hpp"
              DEPENDENCIES monitors
              OPENCV_DEPENDENCIES highgui)

target_link_libraries(object_detection_demo_yolov3_async PRIVATE ngraph::ngraph)


ie_add_sample(NAME object_detection_demo_yolov3_async_multiple_NCS
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/multiple-NCS.cpp"
              HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/object_detection_demo_yolov3_async.hpp"
              DEPENDENCIES monitors
              OPENCV_DEPENDENCIES highgui)

target_link_libraries(object_detection_demo_yolov3_async_multiple_NCS PRIVATE ngraph::ngraph stdc++fs)

ie_add_sample(NAME object_detection_demo_yolov3_async_multiple_NCS_het
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/multiple-NCS-het.cpp"
              HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/object_detection_demo_yolov3_async.hpp"
              DEPENDENCIES monitors
              OPENCV_DEPENDENCIES highgui)

target_link_libraries(object_detection_demo_yolov3_async_multiple_NCS_het PRIVATE ngraph::ngraph stdc++fs)

ie_add_sample(NAME object_detection_demo_yolov3_async_multiple_NCS_het_rr_vis
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/multiple-NCS-het-rr-vis.cpp"
              HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/object_detection_demo_yolov3_async.hpp"
              DEPENDENCIES monitors
              OPENCV_DEPENDENCIES highgui)

target_link_libraries(object_detection_demo_yolov3_async_multiple_NCS_het_rr_vis PRIVATE ngraph::ngraph stdc++fs)

ie_add_sample(NAME object_detection_demo_yolov3_async_multiple_NCS_het_rr
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/multiple-NCS-het-rr.cpp"
              HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/object_detection_demo_yolov3_async.hpp"
              DEPENDENCIES monitors
              OPENCV_DEPENDENCIES highgui)

target_link_libraries(object_detection_demo_yolov3_async_multiple_NCS_het_rr PRIVATE ngraph::ngraph stdc++fs)
