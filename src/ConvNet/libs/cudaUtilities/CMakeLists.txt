cmake_minimum_required (VERSION 2.6)

find_package(CUDA REQUIRED)
  
add_library(cudaUtilities STATIC
    source/cudaUtils/CudaBaseMemoryChunk.cpp
    source/cudaUtils/CudaBaseMemoryChunk.h
    source/cudaUtils/CudaCodeModule.cpp
    source/cudaUtils/CudaCodeModule.h
    source/cudaUtils/CudaDevice.cpp
    source/cudaUtils/CudaDevice.h
    source/cudaUtils/CudaDeviceContext.cpp
    source/cudaUtils/CudaDeviceContext.h
    source/cudaUtils/CudaDeviceMemory.cpp
    source/cudaUtils/CudaDeviceMemory.h
    source/cudaUtils/CudaDriver.cpp
    source/cudaUtils/CudaDriver.h
    source/cudaUtils/CudaFence.cpp
    source/cudaUtils/CudaFence.h
    source/cudaUtils/CudaKernel.cpp
    source/cudaUtils/CudaKernel.h
    source/cudaUtils/CudaPinnedHostMemory.cpp
    source/cudaUtils/CudaPinnedHostMemory.h
    source/cudaUtils/CudaProfilingScope.cpp
    source/cudaUtils/CudaProfilingScope.h
    source/cudaUtils/CudaStream.cpp
    source/cudaUtils/CudaStream.h
#    source/cudaUtils/CudaSurfaceReference.cpp
#    source/cudaUtils/CudaSurfaceReference.h
#    source/cudaUtils/CudaTextureMemory.cpp
#    source/cudaUtils/CudaTextureMemory.h
#    source/cudaUtils/CudaTextureObject.cpp
#    source/cudaUtils/CudaTextureObject.h
#    source/cudaUtils/CudaTextureReference.cpp
#    source/cudaUtils/CudaTextureReference.h
)


target_include_directories(cudaUtilities PUBLIC
    ${CUDA_INCLUDE_DIRS}
)


target_include_directories(cudaUtilities INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/source
)

FIND_LIBRARY(CUDA_LIBRARY cuda /usr/local/cuda/lib64 /usr/lib64/)

target_link_libraries(cudaUtilities PUBLIC 
    ${CUDA_LIBRARY}
)

function(generateInternalCudaKernelImage listOf_PTX_CPP_files sourceFile name options)

    # If CUDA_INCLUDE_DIRS is empty, the FindCMAKE file appends an empty "-I" to the nvcc command line which causes nvcc to fail
    # This workaround switches the command line argument to "-I." which isn't quite right but at least doesn't fail
    if (CUDA_INCLUDE_DIRS)
    else()
    set(CUDA_INCLUDE_DIRS .)
    endif()

    set(CUDA_COMPILE_OPTIONS "-arch compute_30")
    set(CUDA_HOST_COMPILER "g++")
    
    # Invokes the compilation. The resulting .ptx filename is stored in _ptxDstFile
    cuda_compile_ptx(
        _ptxDstFile
        ${sourceFile}
        OPTIONS ${CUDA_COMPILE_OPTIONS} ${options}
    )

    # Extract just the filename without path
    get_filename_component(_filename ${sourceFile} NAME)

    # Package the ptx file in a null terminated C string inside a .ptx.cpp file.
    add_custom_command(OUTPUT ${_filename}.ptx.cpp
        COMMAND ${PROJECT_SOURCE_DIR}/cmake/buildPtxResource.sh ${_filename}.ptx.cpp ${name} ${_ptxDstFile}
        DEPENDS ${_ptxDstFile} ${sourceFile}
    )

    # Append the .ptx.cpp file to the list of .ptx.cpp files that needs to be compiled and linked
    LIST(APPEND ${listOf_PTX_CPP_files} ${_filename}.ptx.cpp)
    set(${listOf_PTX_CPP_files} ${${listOf_PTX_CPP_files}} PARENT_SCOPE)
endfunction()



