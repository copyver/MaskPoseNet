# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yhlever/DeepLearning/AMSMC/tf_ops

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yhlever/DeepLearning/AMSMC/tf_ops/build

# Include any dependencies generated for this target.
include CMakeFiles/tf_sampling.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/tf_sampling.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/tf_sampling.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tf_sampling.dir/flags.make

CMakeFiles/tf_sampling.dir/tf_sampling.cu.o: CMakeFiles/tf_sampling.dir/flags.make
CMakeFiles/tf_sampling.dir/tf_sampling.cu.o: ../tf_sampling.cu
CMakeFiles/tf_sampling.dir/tf_sampling.cu.o: CMakeFiles/tf_sampling.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yhlever/DeepLearning/AMSMC/tf_ops/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/tf_sampling.dir/tf_sampling.cu.o"
	/usr/local/cuda-11.2/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/tf_sampling.dir/tf_sampling.cu.o -MF CMakeFiles/tf_sampling.dir/tf_sampling.cu.o.d -x cu -dc /home/yhlever/DeepLearning/AMSMC/tf_ops/tf_sampling.cu -o CMakeFiles/tf_sampling.dir/tf_sampling.cu.o

CMakeFiles/tf_sampling.dir/tf_sampling.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/tf_sampling.dir/tf_sampling.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/tf_sampling.dir/tf_sampling.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/tf_sampling.dir/tf_sampling.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/tf_sampling.dir/tf_sampling.cpp.o: CMakeFiles/tf_sampling.dir/flags.make
CMakeFiles/tf_sampling.dir/tf_sampling.cpp.o: ../tf_sampling.cpp
CMakeFiles/tf_sampling.dir/tf_sampling.cpp.o: CMakeFiles/tf_sampling.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yhlever/DeepLearning/AMSMC/tf_ops/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/tf_sampling.dir/tf_sampling.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/tf_sampling.dir/tf_sampling.cpp.o -MF CMakeFiles/tf_sampling.dir/tf_sampling.cpp.o.d -o CMakeFiles/tf_sampling.dir/tf_sampling.cpp.o -c /home/yhlever/DeepLearning/AMSMC/tf_ops/tf_sampling.cpp

CMakeFiles/tf_sampling.dir/tf_sampling.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tf_sampling.dir/tf_sampling.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yhlever/DeepLearning/AMSMC/tf_ops/tf_sampling.cpp > CMakeFiles/tf_sampling.dir/tf_sampling.cpp.i

CMakeFiles/tf_sampling.dir/tf_sampling.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tf_sampling.dir/tf_sampling.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yhlever/DeepLearning/AMSMC/tf_ops/tf_sampling.cpp -o CMakeFiles/tf_sampling.dir/tf_sampling.cpp.s

# Object files for target tf_sampling
tf_sampling_OBJECTS = \
"CMakeFiles/tf_sampling.dir/tf_sampling.cu.o" \
"CMakeFiles/tf_sampling.dir/tf_sampling.cpp.o"

# External object files for target tf_sampling
tf_sampling_EXTERNAL_OBJECTS =

CMakeFiles/tf_sampling.dir/cmake_device_link.o: CMakeFiles/tf_sampling.dir/tf_sampling.cu.o
CMakeFiles/tf_sampling.dir/cmake_device_link.o: CMakeFiles/tf_sampling.dir/tf_sampling.cpp.o
CMakeFiles/tf_sampling.dir/cmake_device_link.o: CMakeFiles/tf_sampling.dir/build.make
CMakeFiles/tf_sampling.dir/cmake_device_link.o: CMakeFiles/tf_sampling.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yhlever/DeepLearning/AMSMC/tf_ops/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA device code CMakeFiles/tf_sampling.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tf_sampling.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tf_sampling.dir/build: CMakeFiles/tf_sampling.dir/cmake_device_link.o
.PHONY : CMakeFiles/tf_sampling.dir/build

# Object files for target tf_sampling
tf_sampling_OBJECTS = \
"CMakeFiles/tf_sampling.dir/tf_sampling.cu.o" \
"CMakeFiles/tf_sampling.dir/tf_sampling.cpp.o"

# External object files for target tf_sampling
tf_sampling_EXTERNAL_OBJECTS =

libtf_sampling.so: CMakeFiles/tf_sampling.dir/tf_sampling.cu.o
libtf_sampling.so: CMakeFiles/tf_sampling.dir/tf_sampling.cpp.o
libtf_sampling.so: CMakeFiles/tf_sampling.dir/build.make
libtf_sampling.so: CMakeFiles/tf_sampling.dir/cmake_device_link.o
libtf_sampling.so: CMakeFiles/tf_sampling.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yhlever/DeepLearning/AMSMC/tf_ops/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX shared library libtf_sampling.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tf_sampling.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tf_sampling.dir/build: libtf_sampling.so
.PHONY : CMakeFiles/tf_sampling.dir/build

CMakeFiles/tf_sampling.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tf_sampling.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tf_sampling.dir/clean

CMakeFiles/tf_sampling.dir/depend:
	cd /home/yhlever/DeepLearning/AMSMC/tf_ops/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yhlever/DeepLearning/AMSMC/tf_ops /home/yhlever/DeepLearning/AMSMC/tf_ops /home/yhlever/DeepLearning/AMSMC/tf_ops/build /home/yhlever/DeepLearning/AMSMC/tf_ops/build /home/yhlever/DeepLearning/AMSMC/tf_ops/build/CMakeFiles/tf_sampling.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tf_sampling.dir/depend

