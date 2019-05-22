# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)
	
add_executable(GazeClassifier GazeClassifier.cpp)
target_link_libraries(GazeClassifier LandmarkDetector)
target_link_libraries(GazeClassifier FaceAnalyser)
target_link_libraries(GazeClassifier GazeAnalyser)
target_link_libraries(GazeClassifier Utilities)

install (TARGETS GazeClassifier DESTINATION bin)