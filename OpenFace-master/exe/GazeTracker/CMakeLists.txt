# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)
# Fabian Timm's pupil finding library
include_directories(${EyeLike_SOURCE_DIR}/include)
	
add_executable(GazeTracker GazeTracker.cpp)
target_link_libraries(GazeTracker LandmarkDetector)
target_link_libraries(GazeTracker FaceAnalyser)
target_link_libraries(GazeTracker GazeAnalyser)
target_link_libraries(GazeTracker Utilities)
target_link_libraries(GazeTracker EyeLike)

install (TARGETS GazeTracker DESTINATION bin)