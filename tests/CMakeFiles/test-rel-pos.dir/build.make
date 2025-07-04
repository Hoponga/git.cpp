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
include tests/CMakeFiles/test-rel-pos.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/CMakeFiles/test-rel-pos.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/test-rel-pos.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/test-rel-pos.dir/flags.make

tests/CMakeFiles/test-rel-pos.dir/codegen:
.PHONY : tests/CMakeFiles/test-rel-pos.dir/codegen

tests/CMakeFiles/test-rel-pos.dir/test-rel-pos.c.o: tests/CMakeFiles/test-rel-pos.dir/flags.make
tests/CMakeFiles/test-rel-pos.dir/test-rel-pos.c.o: tests/test-rel-pos.c
tests/CMakeFiles/test-rel-pos.dir/test-rel-pos.c.o: tests/CMakeFiles/test-rel-pos.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/kailashr/ggml/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object tests/CMakeFiles/test-rel-pos.dir/test-rel-pos.c.o"
	cd /Users/kailashr/ggml/tests && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT tests/CMakeFiles/test-rel-pos.dir/test-rel-pos.c.o -MF CMakeFiles/test-rel-pos.dir/test-rel-pos.c.o.d -o CMakeFiles/test-rel-pos.dir/test-rel-pos.c.o -c /Users/kailashr/ggml/tests/test-rel-pos.c

tests/CMakeFiles/test-rel-pos.dir/test-rel-pos.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/test-rel-pos.dir/test-rel-pos.c.i"
	cd /Users/kailashr/ggml/tests && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/kailashr/ggml/tests/test-rel-pos.c > CMakeFiles/test-rel-pos.dir/test-rel-pos.c.i

tests/CMakeFiles/test-rel-pos.dir/test-rel-pos.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/test-rel-pos.dir/test-rel-pos.c.s"
	cd /Users/kailashr/ggml/tests && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/kailashr/ggml/tests/test-rel-pos.c -o CMakeFiles/test-rel-pos.dir/test-rel-pos.c.s

# Object files for target test-rel-pos
test__rel__pos_OBJECTS = \
"CMakeFiles/test-rel-pos.dir/test-rel-pos.c.o"

# External object files for target test-rel-pos
test__rel__pos_EXTERNAL_OBJECTS =

bin/test-rel-pos: tests/CMakeFiles/test-rel-pos.dir/test-rel-pos.c.o
bin/test-rel-pos: tests/CMakeFiles/test-rel-pos.dir/build.make
bin/test-rel-pos: src/libggml.dylib
bin/test-rel-pos: src/libggml-cpu.dylib
bin/test-rel-pos: src/ggml-blas/libggml-blas.dylib
bin/test-rel-pos: src/ggml-metal/libggml-metal.dylib
bin/test-rel-pos: src/libggml-base.dylib
bin/test-rel-pos: tests/CMakeFiles/test-rel-pos.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/kailashr/ggml/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable ../bin/test-rel-pos"
	cd /Users/kailashr/ggml/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test-rel-pos.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/test-rel-pos.dir/build: bin/test-rel-pos
.PHONY : tests/CMakeFiles/test-rel-pos.dir/build

tests/CMakeFiles/test-rel-pos.dir/clean:
	cd /Users/kailashr/ggml/tests && $(CMAKE_COMMAND) -P CMakeFiles/test-rel-pos.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/test-rel-pos.dir/clean

tests/CMakeFiles/test-rel-pos.dir/depend:
	cd /Users/kailashr/ggml && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/kailashr/ggml /Users/kailashr/ggml/tests /Users/kailashr/ggml /Users/kailashr/ggml/tests /Users/kailashr/ggml/tests/CMakeFiles/test-rel-pos.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : tests/CMakeFiles/test-rel-pos.dir/depend

