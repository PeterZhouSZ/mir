cmake_minimum_required (VERSION 2.6)

find_package(ImageMagick COMPONENTS Magick++ REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system thread REQUIRED) 
find_package(TinyXML2 REQUIRED)


generateInternalCudaKernelImage(convnetCudaKernelModulesPtxCppFiles source/convnet/backend/cudaKernels/ConvnetKernelModule.cu PTX_ConvnetKernelModule "")

add_library(convnet STATIC
    source/convnet/backend/cudnn/CuDnnContext.cpp
    source/convnet/backend/cudnn/CuDnnContext.h
    source/convnet/backend/cudnn/CuDnnConvolutionDescriptor.cpp
    source/convnet/backend/cudnn/CuDnnConvolutionDescriptor.h
    source/convnet/backend/cudnn/CuDnnError.cpp
    source/convnet/backend/cudnn/CuDnnError.h
    source/convnet/backend/cudnn/CuDnnFilterDescriptor.cpp
    source/convnet/backend/cudnn/CuDnnFilterDescriptor.h
    source/convnet/backend/cudnn/CuDnnTensorDescriptor.cpp
    source/convnet/backend/cudnn/CuDnnTensorDescriptor.h
    source/convnet/backend/cudnn/CuDnnPoolingDescriptor.cpp
    source/convnet/backend/cudnn/CuDnnPoolingDescriptor.h
    source/convnet/backend/cudnn/CuDnnActivationDescriptor.cpp
    source/convnet/backend/cudnn/CuDnnActivationDescriptor.h
    source/convnet/backend/cudnn/CuDnnDropoutDescriptor.cpp
    source/convnet/backend/cudnn/CuDnnDropoutDescriptor.h
    source/convnet/backend/cudnn/CuDnnLRNDescriptor.cpp
    source/convnet/backend/cudnn/CuDnnLRNDescriptor.h

    source/convnet/backend/layers/ConvolutionLayerImpl_cudnn.cpp
    source/convnet/backend/layers/ConvolutionLayerImpl_cudnn.h
#    source/convnet/backend/layers/TransposedConvolutionLayerImpl_cudnn.cpp
#    source/convnet/backend/layers/TransposedConvolutionLayerImpl_cudnn.h
#    source/convnet/backend/layers/ReluSplitLayerImpl_cudnn.cpp
#    source/convnet/backend/layers/ReluSplitLayerImpl_cudnn.h
    source/convnet/backend/layers/PoolingLayerImpl_cudnn.cpp
    source/convnet/backend/layers/PoolingLayerImpl_cudnn.h
#    source/convnet/backend/layers/UnpoolingLayerImpl_cudnn.cpp
#    source/convnet/backend/layers/UnpoolingLayerImpl_cudnn.h
    source/convnet/backend/layers/InputLayerImpl_cudnn.cpp
    source/convnet/backend/layers/InputLayerImpl_cudnn.h
#    source/convnet/backend/layers/VAEBottleneckLayerImpl_cudnn.cpp
#    source/convnet/backend/layers/VAEBottleneckLayerImpl_cudnn.h
#    source/convnet/backend/layers/ConcatenationLayerImpl_cudnn.cpp
#    source/convnet/backend/layers/ConcatenationLayerImpl_cudnn.h
#    source/convnet/backend/layers/PReluLayerImpl_cudnn.cpp
#    source/convnet/backend/layers/PReluLayerImpl_cudnn.h
    source/convnet/backend/layers/ReluLayerImpl_cudnn.cpp
    source/convnet/backend/layers/ReluLayerImpl_cudnn.h
    source/convnet/backend/layers/DropoutLayerImpl_cudnn.cpp
    source/convnet/backend/layers/DropoutLayerImpl_cudnn.h
    source/convnet/backend/layers/CrossEntropyLossLayerImpl_cudnn.cpp
    source/convnet/backend/layers/CrossEntropyLossLayerImpl_cudnn.h
#    source/convnet/backend/layers/ReconstructionLossLayerImpl_cudnn.cpp
#    source/convnet/backend/layers/ReconstructionLossLayerImpl_cudnn.h
#    source/convnet/backend/layers/ElemWiseOpLayerImpl_cudnn.cpp
#    source/convnet/backend/layers/ElemWiseOpLayerImpl_cudnn.h
#    source/convnet/backend/layers/GradientNegatingLayerImpl_cudnn.cpp
#    source/convnet/backend/layers/GradientNegatingLayerImpl_cudnn.h
#    source/convnet/backend/layers/GANLossLayerImpl_cudnn.cpp
#    source/convnet/backend/layers/GANLossLayerImpl_cudnn.h
#    source/convnet/backend/layers/SaturationLayerImpl_cudnn.cpp
#    source/convnet/backend/layers/SaturationLayerImpl_cudnn.h
#    source/convnet/backend/layers/InstanceNormalizationLayerImpl_cudnn.cpp
#    source/convnet/backend/layers/InstanceNormalizationLayerImpl_cudnn.h
#    source/convnet/backend/layers/LocalResponseNormalizationLayerImpl_cudnn.cpp
#    source/convnet/backend/layers/LocalResponseNormalizationLayerImpl_cudnn.h
#    source/convnet/backend/layers/CropLayerImpl_cudnn.cpp
#    source/convnet/backend/layers/CropLayerImpl_cudnn.h
#    source/convnet/backend/layers/BatchRenormLayerImpl_cudnn.cpp
#    source/convnet/backend/layers/BatchRenormLayerImpl_cudnn.h
    source/convnet/backend/layers/BatchNormLayerImpl_cudnn.cpp
    source/convnet/backend/layers/BatchNormLayerImpl_cudnn.h
    source/convnet/backend/layers/TripletLossLayerImpl_cudnn.cpp
    source/convnet/backend/layers/TripletLossLayerImpl_cudnn.h

    source/convnet/backend/ConvNetImpl_cudnn.cpp
    source/convnet/backend/ConvNetImpl_cudnn.h
    source/convnet/backend/CudnnAuxKernels.cpp
    source/convnet/backend/CudnnAuxKernels.h
    source/convnet/backend/CuDnnWrapper.cpp
    source/convnet/backend/CuDnnWrapper.h
    source/convnet/backend/ExecutionWorkspaceImpl_cudnn.cpp
    source/convnet/backend/ExecutionWorkspaceImpl_cudnn.h
    source/convnet/backend/ExecutionStreamImpl_cudnn.cpp
    source/convnet/backend/ExecutionStreamImpl_cudnn.h
    source/convnet/backend/TensorImpl_cudnn.cpp
    source/convnet/backend/TensorImpl_cudnn.h
    source/convnet/backend/PinnedMemoryAllocator_cudnn.cpp
    source/convnet/backend/PinnedMemoryAllocator_cudnn.h

    source/convnet/layers/CrossEntropyLossLayer.cpp
    source/convnet/layers/CrossEntropyLossLayer.h
#    source/convnet/layers/PReluLayer.h
#    source/convnet/layers/PReluLayer.cpp
    source/convnet/layers/ReluLayer.h
    source/convnet/layers/ReluLayer.cpp
#    source/convnet/layers/ConcatenationLayer.h
#    source/convnet/layers/ConcatenationLayer.cpp
#    source/convnet/layers/VAEBottleneckLayer.h
#    source/convnet/layers/VAEBottleneckLayer.cpp
    source/convnet/layers/ConvolutionLayer.cpp
    source/convnet/layers/ConvolutionLayer.h
#    source/convnet/layers/TransposedConvolutionLayer.cpp
#    source/convnet/layers/TransposedConvolutionLayer.h
#    source/convnet/layers/ReluSplitLayer.cpp
#    source/convnet/layers/ReluSplitLayer.h
    source/convnet/layers/PoolingLayer.cpp
    source/convnet/layers/PoolingLayer.h
#    source/convnet/layers/UnpoolingLayer.cpp
#    source/convnet/layers/UnpoolingLayer.h
    source/convnet/layers/InputLayer.cpp
    source/convnet/layers/InputLayer.h
    source/convnet/layers/Layer.h
    source/convnet/layers/Layer.cpp
    source/convnet/layers/DropoutLayer.h
    source/convnet/layers/DropoutLayer.cpp
#    source/convnet/layers/ReconstructionLossLayer.cpp
#    source/convnet/layers/ReconstructionLossLayer.h
#    source/convnet/layers/ElemWiseOpLayer.cpp
#    source/convnet/layers/ElemWiseOpLayer.h
#    source/convnet/layers/GradientNegatingLayer.cpp
#    source/convnet/layers/GradientNegatingLayer.h
#    source/convnet/layers/GANLossLayer.cpp
#    source/convnet/layers/GANLossLayer.h
#    source/convnet/layers/SaturationLayer.cpp
#    source/convnet/layers/SaturationLayer.h
#    source/convnet/layers/InstanceNormalizationLayer.cpp
#    source/convnet/layers/InstanceNormalizationLayer.h
#    source/convnet/layers/LocalResponseNormalizationLayer.cpp
#    source/convnet/layers/LocalResponseNormalizationLayer.h
#    source/convnet/layers/CropLayer.cpp
#    source/convnet/layers/CropLayer.h
#    source/convnet/layers/BatchRenormLayer.cpp
#    source/convnet/layers/BatchRenormLayer.h
    source/convnet/layers/BatchNormLayer.cpp
    source/convnet/layers/BatchNormLayer.h
    source/convnet/layers/TripletLossLayer.cpp
    source/convnet/layers/TripletLossLayer.h

    source/convnet/reporting/HTTPReport.cpp
    source/convnet/reporting/HTTPReport.h
    source/convnet/reporting/TensorRenderer.cpp
    source/convnet/reporting/TensorRenderer.h
#    source/convnet/reporting/ActivationAnalysis.cpp
#    source/convnet/reporting/ActivationAnalysis.h
    source/convnet/reporting/State2Html.h
    source/convnet/reporting/State2Html.cpp

    source/convnet/Blobb.cpp
    source/convnet/Blobb.h
    source/convnet/FileBlob.cpp
    source/convnet/FileBlob.h
    source/convnet/Convnet.cpp
    source/convnet/Convnet.h
    source/convnet/ExecutionWorkspace.cpp
    source/convnet/ExecutionWorkspace.h
    source/convnet/ExecutionStream.cpp
    source/convnet/ExecutionStream.h
    source/convnet/NetworkState.cpp
    source/convnet/NetworkState.h
    source/convnet/Tensor.cpp
    source/convnet/Tensor.h
    source/convnet/FilterRenderer.cpp
    source/convnet/FilterRenderer.h

#    source/convnet/training/CocoHelpers.cpp
#    source/convnet/training/CocoHelpers.h
    source/convnet/training/TrainingDataSource.cpp
    source/convnet/training/TrainingDataSource.h
    source/convnet/training/WholeImageClassificationTrainer.cpp
    source/convnet/training/WholeImageClassificationTrainer.h

#    source/convnet/training/VAETrainer.cpp
#    source/convnet/training/VAETrainer.h
#    source/convnet/training/RegressionTrainer.cpp
#    source/convnet/training/RegressionTrainer.h
    source/convnet/training/Trainer.cpp
    source/convnet/training/Trainer.h
#    source/convnet/training/ClassificationTrainer.cpp
#    source/convnet/training/ClassificationTrainer.h

    ${convnetCudaKernelModulesPtxCppFiles}
)

set(data_files 
    data/reporting/fonts/glyphicons-halflings-regular.woff2
    data/reporting/fonts/glyphicons-halflings-regular.ttf
    data/reporting/fonts/glyphicons-halflings-regular.woff
    data/reporting/fonts/glyphicons-halflings-regular.svg
    data/reporting/fonts/glyphicons-halflings-regular.eot
    data/reporting/css/vis-network.min.css
    data/reporting/css/style.css
    data/reporting/css/bootstrap.min.css
    data/reporting/js/vis.js
    data/reporting/js/bootstrap.min.js
    data/reporting/js/plotly-latest.min.js
    data/reporting/js/jquery-3.2.1.min.js
    data/reporting/js/utils.js
)

foreach(file ${data_files})
    configure_file(${file} ${PROJECT_BINARY_DIR}/${file} COPYONLY)
endforeach(file)

if (${ImageMagick_LIBRARIES} MATCHES "Q16")
target_compile_definitions(convnet PUBLIC "MAGICKCORE_QUANTUM_DEPTH=16")
else()
target_compile_definitions(convnet PUBLIC "MAGICKCORE_QUANTUM_DEPTH=8")
endif()

if (${ImageMagick_LIBRARIES} MATCHES "HDRI")
target_compile_definitions(convnet PUBLIC "MAGICKCORE_HDRI_ENABLE=true")
else()
target_compile_definitions(convnet PUBLIC "MAGICKCORE_HDRI_ENABLE=false")
endif()


target_include_directories(convnet PUBLIC
    ${ImageMagick_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${TINYXML2_INCLUDE_DIR}
)

target_include_directories(convnet INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/source
)

target_link_libraries(convnet PUBLIC
    ${ImageMagick_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    ${Boost_LIBRARIES}
    ${TINYXML2_LIBRARY}
	pthread
)


target_link_libraries(convnet PUBLIC
    common
    cudaUtilities
#    miscExtern
    -L/usr/local/cuda/lib64 -lcudnn
)

add_subdirectory(unitTests/)
