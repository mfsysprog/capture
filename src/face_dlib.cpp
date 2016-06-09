// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.


    This example is essentially just a version of the face_landmark_detection_ex.cpp
    example modified to use OpenCV's VideoCapture object to read from a camera instead
    of files.


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.
*/

#include <dlib/opencv.h>
//#include <opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <iostream>
#include <dlib/gui_widgets.h>


using namespace dlib;
using namespace std;

void draw_polyline(cv::Mat &img, const dlib::full_object_detection& d, const int start, const int end, bool isClosed = false)
{
    std::vector <cv::Point> points;
    for (int i = start; i <= end; ++i)
    {
        points.push_back(cv::Point(d.part(i).x(), d.part(i).y()));
    }
    cv::polylines(img, points, isClosed, cv::Scalar(255,0,0), 2, 16);

}

void render_face (cv::Mat &img, const dlib::full_object_detection& d)
{
    DLIB_CASSERT
    (
     d.num_parts() == 68,
     "\n\t Invalid inputs were given to this function. "
     << "\n\t d.num_parts():  " << d.num_parts()
     );

    draw_polyline(img, d, 0, 16);           // Jaw line
    draw_polyline(img, d, 17, 21);          // Left eyebrow
    draw_polyline(img, d, 22, 26);          // Right eyebrow
    draw_polyline(img, d, 27, 30);          // Nose bridge
    draw_polyline(img, d, 30, 35, true);    // Lower nose
    draw_polyline(img, d, 36, 41, true);    // Left eye
    draw_polyline(img, d, 42, 47, true);    // Right Eye
    draw_polyline(img, d, 48, 59, true);    // Outer lip
    draw_polyline(img, d, 60, 67, true);    // Inner lip

}

int main(int argc, char** argv)
{
    #define FACE_DOWNSAMPLE_RATIO 2

	try
    {
		if (argc != 2)
		        {
		            cout << "Call this program with a number 0 or 1 to indicate the /dev/video(x) input to use." << endl;
		            return 0;
		        }

		std::vector<std::string> args(argv, argv+argc);
		cv::VideoCapture cap;

		if (args[1] == "0")
			{
				cap.open(0);
			}
		else if (args[1] == "1")
			{
				cap.open(1);
			}
		else {
		    cout << "Invalid cam number specified." << endl;
			return 0;
		}
        image_window win_faces;

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        cout << "Reading in shape predictor..." << endl;
        deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
        cout << "Done reading in shape predictor ..." << endl;

        std::vector<rectangle> faces;
        cv::Mat im;
        cv::Mat im_small;

        // Grab and process frames until the main window is closed by the user.
        	cout << "Grabbing a frame..." << endl;
        	cap.set(CV_CAP_PROP_FRAME_WIDTH,1280);   // width pixels
        	cap.set(CV_CAP_PROP_FRAME_HEIGHT,720);   // height pixels
        	if(!cap.isOpened()){   // connect to the camera
        	         cout << "Failed to connect to the camera." << endl;
        	         return 1;
            }
            // Grab a frame
            cap >> im;
            // Release the cam
            cap.release();

            cout << "Image size: " << im.cols << " x " << im.rows << endl;

            // Resize image for face detection
            cv::resize(im, im_small, cv::Size(), 1.0/FACE_DOWNSAMPLE_RATIO, 1.0/FACE_DOWNSAMPLE_RATIO);
            // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
            // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
            // long as temp is valid.  Also don't do anything to temp that would cause it
            // to reallocate the memory which stores the image as that will make cimg
            // contain dangling pointers.  This basically means you shouldn't modify temp
            // while using cimg.
            //cv_image<bgr_pixel> cimg(temp);
            //array2d<rgb_pixel> cimg;
            //assign_image(cimg, cv_image<bgr_pixel>(im));
            cv_image<bgr_pixel> cimg_small(im_small);
            cv_image<bgr_pixel> cimg(im);
            //array2d<rgb_pixel> c2img;
            //assign_image(c2img, cv_image<bgr_pixel>(cimg));
            // Make the image larger so we can detect small faces.
            //pyramid_up(cimg);

            cout << "Detecting faces..." << endl;
            // Detect faces
            faces = detector(cimg_small);
            // Find the pose of each face.
            std::vector<full_object_detection> shapes;
            if (faces.size() == 0)
            {
            	cout << "No faces detected." << endl;
            	return 0;
            }
            for (unsigned long i = 0; i < faces.size(); ++i)
            {
            	// Resize obtained rectangle for full resolution image.
            	     rectangle r(
            	                   (long)(faces[i].left() * FACE_DOWNSAMPLE_RATIO),
            	                   (long)(faces[i].top() * FACE_DOWNSAMPLE_RATIO),
            	                   (long)(faces[i].right() * FACE_DOWNSAMPLE_RATIO),
            	                   (long)(faces[i].bottom() * FACE_DOWNSAMPLE_RATIO)
            	                );

            	// Landmark detection on full sized image
            	full_object_detection shape = pose_model(cimg, r);
                //shapes.push_back(pose_model(cimg, faces[i]));
                shapes.push_back(shape);
                cout << "Displaying on screen..." << endl;
                // Custom Face Render
                render_face(im, shape);
            }

            // Display it all on the screen
            //win.clear_overlay();
            //win.set_image(cimg);
            //win.add_overlay(render_face_detections(shapes));
            // We can also extract copies of each face that are cropped, rotated upright,
            // and scaled to a standard size as shown here:
            dlib::array<array2d<rgb_pixel> > face_chips;
            extract_image_chips(cimg, get_face_chip_details(shapes), face_chips);
            win_faces.set_image(tile_images(face_chips));

            while(!win_faces.is_closed()){}
    }
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}


