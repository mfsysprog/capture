################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/face_dlib.cpp \
/home/erik/dlib-18.18/dlib/all/source.cpp 

OBJS += \
./src/face_dlib.o \
./src/source.o 

CPP_DEPS += \
./src/face_dlib.d \
./src/source.d 


# Each subdirectory must supply rules for building sources it contributes
src/face_dlib.o: ../src/face_dlib.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	arm-linux-gnueabihf-g++ -I/usr/include/arm-linux-gnueabihf/c++/4.9 -I/home/erik/dlib-18.18 -I/home/erik/opencv-3.1.0/modules/core/include -I/home/erik/opencv-3.1.0/modules/core/include/opencv2 -I/home/erik/opencv-3.1.0/include/opencv -I/home/erik/opencv-3.1.0/include/opencv2 -I/home/erik/opencv-3.1.0/modules/highgui/include -I/home/erik/opencv-3.1.0/modules/imgproc/include -I/home/erik/opencv-3.1.0/modules/photo/include -I/home/erik/opencv-3.1.0/modules/imgcodecs/include -I/home/erik/opencv-3.1.0/modules/objdetect/include -I/home/erik/opencv-3.1.0/modules/videoio/include -O3 -Wall -c -fmessage-length=0 -v -MMD -MP -MF"$(@:%.o=%.d)" -MT"src/face_dlib.d" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/source.o: /home/erik/dlib-18.18/dlib/all/source.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	arm-linux-gnueabihf-g++ -I/home/erik/dlib-18.18 -I/home/erik/opencv-3.1.0/modules/core/include -I/home/erik/opencv-3.1.0/modules/core/include/opencv2 -I/home/erik/opencv-3.1.0/include/opencv -I/home/erik/opencv-3.1.0/include/opencv2 -I/home/erik/opencv-3.1.0/modules/photo/include -I/home/erik/opencv-3.1.0/modules/highgui/include -I/home/erik/opencv-3.1.0/modules/videoio/include -I/home/erik/opencv-3.1.0/modules/objdetect/include -I/home/erik/opencv-3.1.0/modules/imgproc/include -I/home/erik/opencv-3.1.0/modules/imgcodecs/include -O3 -Wall -c -fmessage-length=0 -v -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


