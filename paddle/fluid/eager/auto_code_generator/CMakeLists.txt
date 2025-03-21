add_subdirectory(generator)

set(EAGER_GENERATOR_DEPS
    ${GLOB_OP_LIB}
    ${GLOB_OPERATOR_DEPS}
    pybind
    proto_desc
    executor
    layer
    tracer
    engine
    imperative_profiler
    imperative_flag)

if(WITH_CINN)
  list(REMOVE_ITEM EAGER_GENERATOR_DEPS imperative_flag)
endif()

if(WITH_CUSTOM_DEVICE)
  set(EAGER_GENERATOR_DEPS ${EAGER_GENERATOR_DEPS}
                           custom_device_common_op_registry)
endif()

add_executable(eager_generator eager_generator.cc)
target_link_libraries(eager_generator ${EAGER_GENERATOR_DEPS})

get_property(os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
target_link_libraries(eager_generator ${os_dependency_modules})

if(WITH_ROCM)
  target_link_libraries(eager_generator ${ROCM_HIPRTC_LIB})
endif()

if(WITH_CINN)
  target_link_libraries(eager_generator ${PYTHON_LIBRARIES})
endif()

# Prepare file structure
message(
  "Generate dygraph file structure at path: ${PADDLE_SOURCE_DIR}/paddle/fluid/eager/generated"
)

set(CODE_GEN_SPLIT_FILE_COUNT "8")

execute_process(
  COMMAND
    "${PYTHON_EXECUTABLE}"
    "${PADDLE_SOURCE_DIR}/paddle/fluid/eager/auto_code_generator/generate_file_structures.py"
    "${PADDLE_SOURCE_DIR}/paddle/fluid/eager/" "${CODE_GEN_SPLIT_FILE_COUNT}")

if(WIN32)
  set(EAGER_CODEGEN_DEPS eager_generator)
  if("${CMAKE_GENERATOR}" STREQUAL "Ninja")
    set(eager_generator_path "${CMAKE_CURRENT_BINARY_DIR}")
  else()
    set(eager_generator_path "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}")
  endif()

  if(WITH_SHARED_PHI)
    message("Copied phi.dll for Eager AutoCodeGen")
    add_custom_command(
      OUTPUT ${eager_generator_path}/phi.dll
      COMMAND ${CMAKE_COMMAND} -E copy ${PHI_LIB} ${eager_generator_path}
      DEPENDS phi)
    list(APPEND EAGER_CODEGEN_DEPS ${eager_generator_path}/phi.dll)
  endif()

  add_custom_command(
    OUTPUT ${eager_generator_path}/common.dll
    COMMAND ${CMAKE_COMMAND} -E copy ${COMMON_LIB} ${eager_generator_path}
    DEPENDS common)
  list(APPEND EAGER_CODEGEN_DEPS ${eager_generator_path}/common.dll)

  if(WITH_SHARED_IR)
    add_custom_command(
      OUTPUT ${eager_generator_path}/ir.dll
      COMMAND ${CMAKE_COMMAND} -E copy ${IR_LIB} ${eager_generator_path}
      DEPENDS pir)
    list(APPEND EAGER_CODEGEN_DEPS ${eager_generator_path}/ir.dll)
  endif()

  if(${CBLAS_PROVIDER} STREQUAL MKLML)
    message("Copied libiomp5md.dll for Eager AutoCodeGen")
    add_custom_command(
      OUTPUT ${eager_generator_path}/libiomp5md.dll
      COMMAND ${CMAKE_COMMAND} -E copy ${MKLML_SHARED_IOMP_LIB}
              ${eager_generator_path}
      DEPENDS mklml)
    list(APPEND EAGER_CODEGEN_DEPS ${eager_generator_path}/libiomp5md.dll)
  else()
    message("Copied openblas.dll for Eager AutoCodeGen")
    add_custom_command(
      OUTPUT ${eager_generator_path}/openblas.dll
      COMMAND ${CMAKE_COMMAND} -E copy ${OPENBLAS_SHARED_LIB}
              ${eager_generator_path}
      DEPENDS extern_openblas)
    list(APPEND EAGER_CODEGEN_DEPS ${eager_generator_path}/openblas.dll)
  endif()

  if(WITH_ONEDNN)
    message("Copied mkldnn.dll for Eager AutoCodeGen")
    add_custom_command(
      OUTPUT ${eager_generator_path}/mkldnn.dll
      COMMAND ${CMAKE_COMMAND} -E copy ${ONEDNN_SHARED_LIB}
              ${eager_generator_path}
      DEPENDS onednn)
    list(APPEND EAGER_CODEGEN_DEPS ${eager_generator_path}/mkldnn.dll)
  endif()

  if(WITH_ONNXRUNTIME)
    message("Copied onnxruntime for Eager AutoCodeGen")
    add_custom_command(
      OUTPUT ${eager_generator_path}/onnxruntime.dll
      COMMAND ${CMAKE_COMMAND} -E copy ${ONNXRUNTIME_SHARED_LIB}
              ${eager_generator_path}
      DEPENDS onnxruntime)
    list(APPEND EAGER_CODEGEN_DEPS ${eager_generator_path}/onnxruntime.dll)
    add_custom_command(
      OUTPUT ${eager_generator_path}/paddle2onnx.dll
      COMMAND ${CMAKE_COMMAND} -E copy ${PADDLE2ONNX_LIB}
              ${eager_generator_path}
      DEPENDS paddle2onnx)
    list(APPEND EAGER_CODEGEN_DEPS ${eager_generator_path}/paddle2onnx.dll)
  endif()

  add_custom_target(
    legacy_eager_codegen
    COMMAND
      "${eager_generator_path}/eager_generator.exe"
      "${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/fluid_generated"
      "${CODE_GEN_SPLIT_FILE_COUNT}"
    DEPENDS ${EAGER_CODEGEN_DEPS}
    VERBATIM)
else()
  add_custom_target(
    legacy_eager_codegen
    COMMAND
      ${CMAKE_COMMAND} -E env
      "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${CMAKE_CURRENT_BINARY_DIR}/../../pybind:${PADDLE_BINARY_DIR}/third_party/install/mklml/lib"
      "${CMAKE_CURRENT_BINARY_DIR}/eager_generator"
      "${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/fluid_generated"
      "${CODE_GEN_SPLIT_FILE_COUNT}"
    DEPENDS eager_generator
    VERBATIM)
endif()
