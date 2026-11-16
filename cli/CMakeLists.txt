# nxr-neuro: the one command line. Every `nxr-io` verb (dataset · subject · surface · convert · read) and the pipelines (`eigen solve`, `noisecov compute`, `inverse compute`).
add_executable(nxr-neuro main.cpp registry.cpp output.cpp command_table.cpp nxr_neuro.cpp read_command.cpp convert_command.cpp)
target_link_libraries(nxr-neuro PRIVATE $<IF:$<TARGET_EXISTS:nxr_pipelines>,nxr_pipelines,nxr_io>)
target_compile_definitions(nxr-neuro PRIVATE NXR_NEURO_VERSION="${PROJECT_VERSION}")
target_include_directories(nxr-neuro PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/extern)   # CLI11 (vendored, BSD-3-Clause)
if(TARGET nxr_pipelines)
  target_sources(nxr-neuro PRIVATE eigen_command.cpp meeg_command.cpp speedspectrum_command.cpp measure_command.cpp run_command.cpp)
  target_compile_definitions(nxr-neuro PRIVATE NXR_NEURO_WITH_PIPELINES)
endif()
if(MSVC)
  target_compile_options(nxr-neuro PRIVATE /W4)
else()
  target_compile_options(nxr-neuro PRIVATE -Wall -Wextra)
endif()

# the man pages and the shell completion, made from the command table (cli/command_table.cpp) and installed with the command
add_custom_target(nxr-neuro-docs ALL
  COMMAND nxr-neuro __man ${CMAKE_BINARY_DIR}/share/man/man1
  COMMAND nxr-neuro __completions ${CMAKE_BINARY_DIR}/share/completions
  DEPENDS nxr-neuro COMMENT "nxr-neuro: man pages and shell completion")
install(TARGETS nxr-neuro RUNTIME DESTINATION bin)
install(DIRECTORY ${CMAKE_BINARY_DIR}/share/man/ DESTINATION share/man)
install(FILES ${CMAKE_BINARY_DIR}/share/completions/nxr-neuro.bash DESTINATION share/bash-completion/completions RENAME nxr-neuro)
install(FILES ${CMAKE_BINARY_DIR}/share/completions/_nxr-neuro DESTINATION share/zsh/site-functions)
install(FILES ${CMAKE_BINARY_DIR}/share/completions/nxr-neuro.fish DESTINATION share/fish/vendor_completions.d)

# the command line, tested through the built executable (cli/test/cli_tests.cmake); the cases need the pipelines
if(NXR_NEURO_BUILD_TESTS AND TARGET nxr_pipelines)
  foreach(case table help verbs aliases range output json generated)
    add_test(NAME cli_${case} COMMAND ${CMAKE_COMMAND} -DCLI=$<TARGET_FILE:nxr-neuro> -DWORK=${CMAKE_CURRENT_BINARY_DIR}/test-${case} -DCASE=${case}
                                      -P ${CMAKE_CURRENT_SOURCE_DIR}/test/cli_tests.cmake)
  endforeach()
endif()
