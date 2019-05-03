cmake_minimum_required (VERSION 2.6)
   
find_package(Boost COMPONENTS system thread filesystem REQUIRED)

add_executable(andysSandbox
	source/main.cpp
#	source/PlotMovement.cpp
	#source/CategoryClassification.cpp
#	source/ImageClassification.cpp
	source/ImageClassificationMatrixPlots.cpp
#	source/ViewRecallMatching.cpp
	#source/RawScanPathDataSource.cpp
#	source/RawScanPathActivationAnalysis.cpp
    source/HistogramDataSource.cpp
#	source/ViewRecallMatchingHistogram.cpp
#    source/ViewRecallMatchingHistogramTripletLoss.cpp
    source/ViewRecallMatchingHistogramTripletLossCrossVal.cpp
    source/ViewRecallMatchingHistogramTripletLossCrossValIndividual.cpp
#    source/ViewRecallMatchingHistogramTripletLossTraining.cpp
    source/ViewRecallMatchingHistogramTripletLossExportDescriptors.cpp
)

target_include_directories(andysSandbox PUBLIC
    ${Boost_INCLUDE_DIRS}
)


target_link_libraries(andysSandbox
	common
    cudaUtilities
	convnet
	eyeTracking
	-lnvToolsExt
    -lcudart
    ${Boost_LIBRARIES}
)

