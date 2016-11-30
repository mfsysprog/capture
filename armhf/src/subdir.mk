################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/thread.cpp 

OBJS += \
./src/thread.o 

CPP_DEPS += \
./src/thread.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	arm-linux-gnueabihf-g++ -std=c++14 -I/home/erik/opencv-3.1.0/modules/core/include -I/home/erik/opencv-3.1.0/modules/core/include/opencv2 -I/home/erik/opencv-3.1.0/include/opencv -I/home/erik/opencv-3.1.0/include/opencv2 -I/home/erik/opencv-3.1.0/modules/imgproc/include -I/home/erik/opencv-3.1.0/modules/features2d/include -I/home/erik/opencv-3.1.0/modules/calib3d/include -I/home/erik/opencv-3.1.0/modules/photo/include -I/home/erik/opencv-3.1.0/modules/highgui/include -I/home/erik/opencv-3.1.0/modules/video/include -I/home/erik/opencv-3.1.0/modules/videoio/include -I/home/erik/opencv-3.1.0/modules/objdetect/include -I/home/erik/opencv-3.1.0/modules/imgcodecs/include -I/home/erik/dlib-18.18 -O0 -g3 -Wall -c -fmessage-length=0 -mcpu=cortex-a8 -v -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


