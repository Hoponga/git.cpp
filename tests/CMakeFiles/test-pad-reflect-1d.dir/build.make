# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 4.0

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/kailashr/ggml

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/kailashr/ggml

# Include any dependencies generated for this target.
include tests/CMakeFiles/test-pad-reflect-1d.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/CMakeFiles/test-pad-reflect-1d.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/test-pad-reflect-1d.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/test-pad-reflect-1d.dir/flags.make

tests/CMakeFiles/test-pad-reflect-1d.dir/codegen:
.PHONY : tests/CMakeFiles/test-pad-reflect-1d.dir/codegen

tests/CMakeFiles/test-pad-reflect-1d.dir/test-pad-reflect-1d.cpp.o: tests/CMakeFiles/test-pad-reflect-1d.dir/flags.make
tests/CMakeFiles/test-pad-reflect-1d.dir/test-pad-reflect-1d.cpp.o: tests/test-pad-reflect-1d.cpp
tests/CMakeFiles/test-pad-reflect-1d.dir/test-pad-reflect-1d.cpp.o: tests/CMakeFiles/test-pad-reflect-1d.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/kailashr/ggml/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/CMakeFiles/test-pad-reflect-1d.dir/test-pad-reflect-1d.cpp.o"
	cd /Users/kailashr/ggml/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/CMakeFiles/test-pad-reflect-1d.dir/test-pad-reflect-1d.cpp.o -MF CMakeFiles/test-pad-reflect-1d.dir/test-pad-reflect-1d.cpp.o.d -o CMakeFiles/test-pad-reflect-1d.dir/test-pad-reflect-1d.cpp.o -c /Users/kailashr/ggml/tests/test-pad-reflect-1d.cpp

tests/CMakeFiles/test-pad-reflect-1d.dir/test-pad-reflect-1d.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/test-pad-reflect-1d.dir/test-pad-reflect-1d.cpp.i"
	cd /Users/kailashr/ggml/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/kailashr/ggml/tests/test-pad-reflect-1d.cpp > CMakeFiles/test-pad-reflect-1d.dir/test-pad-reflect-1d.cpp.i

tests/CMakeFiles/test-pad-reflect-1d.dir/test-pad-reflect-1d.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/test-pad-reflect-1d.dir/test-pad-reflect-1d.cpp.s"
	cd /Users/kailashr/ggml/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/kailashr/ggml/tests/test-pad-reflect-1d.cpp -o CMakeFiles/test-pad-reflect-1d.dir/test-pad-reflect-1d.cpp.s

# Object files for target test-pad-reflect-1d
test__pad__reflect__1d_OBJECTS = \
"CMakeFiles/test-pad-reflect-1d.dir/test-pad-reflect-1d.cpp.o"

# External object files for target test-pad-reflect-1d
test__pad__reflect__1d_EXTERNAL_OBJECTS =

bin/test-pad-reflect-1d: tests/CMakeFiles/test-pad-reflect-1d.dir/test-pad-reflect-1d.cpp.o
bin/test-pad-reflect-1d: tests/CMakeFiles/test-pad-reflect-1d.dir/build.make
bin/test-pad-reflect-1d: src/libggml.dylib
bin/test-pad-reflect-1d: src/libggml-cpu.dylib
bin/test-pad-reflect-1d: src/ggml-blas/libggml-blas.dylib
bin/test-pad-reflect-1d: src/ggml-metal/libggml-metal.dylib
bin/test-pad-reflect-1d: src/libggml-base.dylib
bin/test-pad-reflect-1d: tests/CMakeFiles/test-pad-reflect-1d.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/kailashr/ggml/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/test-pad-reflect-1d"
	cd /Users/kailashr/ggml/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test-pad-reflect-1d.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/test-pad-reflect-1d.dir/build: bin/test-pad-reflect-1d
.PHONY : tests/CMakeFiles/test-pad-reflect-1d.dir/build

tests/CMakeFiles/test-pad-reflect-1d.dir/clean:
	cd /Users/kailashr/ggml/tests && $(CMAKE_COMMAND) -P CMakeFiles/test-pad-reflect-1d.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/test-pad-reflect-1d.dir/clean

tests/CMakeFiles/test-pad-reflect-1d.dir/depend:
	cd /Users/kailashr/ggml && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/kailashr/ggml /Users/kailashr/ggml/tests /Users/kailashr/ggml /Users/kailashr/ggml/tests /Users/kailashr/ggml/tests/CMakeFiles/test-pad-reflect-1d.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : tests/CMakeFiles/test-pad-reflect-1d.dir/depend

