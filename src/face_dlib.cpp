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

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <iostream>
#include <stdlib.h>
#include <fstream>

using namespace dlib;
using namespace std;
using namespace cv;

void draw_polyline(cv::Mat &img, const dlib::full_object_detection& d, const int start, const int end, bool isClosed = false)
{
    std::vector <cv::Point> points;
    for (int i = start; i <= end; ++i)
    {
        points.push_back(cv::Point(d.part(i).x(), d.part(i).y()));
    }
    cv::polylines(img, points, isClosed, cv::Scalar(255,0,0), 2, 16);
}

//Read points from text file
std::vector<Point2f> readPoints(string pointsFileName){
	std::vector<Point2f> points;
	std::fstream inFile;
	inFile.open(pointsFileName.c_str(), std::ios::in );
    float x, y;
    while(inFile >> x >> y)
    {
        points.push_back(Point2f(x,y));

    }

	return points;
}

// Apply affine transform calculated using srcTri and dstTri to src
void applyAffineTransform(Mat &warpImage, Mat &src, std::vector<Point2f> &srcTri, std::vector<Point2f> &dstTri)
{
    // Given a pair of triangles, find the affine transform.
    Mat warpMat = getAffineTransform( srcTri, dstTri );

    // Apply the Affine Transform just found to the src image
    warpAffine( src, warpImage, warpMat, warpImage.size(), INTER_LINEAR, BORDER_REFLECT_101);
}


// Calculate Delaunay triangles for set of points
// Returns the vector of indices of 3 points for each triangle
static void calculateDelaunayTriangles(Rect rect, std::vector<Point2f> &points, std::vector< std::vector<int> > &delaunayTri){

	// Create an instance of Subdiv2D
    Subdiv2D subdiv(rect);

	// Insert points into subdiv
    for( std::vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
        subdiv.insert(*it);

	std::vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	std::vector<Point2f> pt(3);
	std::vector<int> ind(3);

	for( size_t i = 0; i < triangleList.size(); i++ )
	{
		Vec6f t = triangleList[i];
		pt[0] = Point2f(t[0], t[1]);
		pt[1] = Point2f(t[2], t[3]);
		pt[2] = Point2f(t[4], t[5 ]);

		if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])){
			for(int j = 0; j < 3; j++)
				for(size_t k = 0; k < points.size(); k++)
					if(abs(pt[j].x - points[k].x) < 1.0 && abs(pt[j].y - points[k].y) < 1)
						ind[j] = k;

			delaunayTri.push_back(ind);
		}
	}

}


// Warps and alpha blends triangular regions from img1 and img2 to img
void warpTriangle(Mat &img1, Mat &img2, std::vector<Point2f> &t1, std::vector<Point2f> &t2)
{

    Rect r1 = boundingRect(t1);
    Rect r2 = boundingRect(t2);

    // Offset points by left top corner of the respective rectangles
    std::vector<Point2f> t1Rect, t2Rect;
    std::vector<Point> t2RectInt;
    for(int i = 0; i < 3; i++)
    {

        t1Rect.push_back( Point2f( t1[i].x - r1.x, t1[i].y -  r1.y) );
        t2Rect.push_back( Point2f( t2[i].x - r2.x, t2[i].y - r2.y) );
        t2RectInt.push_back( Point(t2[i].x - r2.x, t2[i].y - r2.y) ); // for fillConvexPoly

    }

    // Get mask by filling triangle
    Mat mask = Mat::zeros(r2.height, r2.width, CV_32FC3);
    fillConvexPoly(mask, t2RectInt, Scalar(1.0, 1.0, 1.0), 16, 0);

    // Apply warpImage to small rectangular patches
    Mat img1Rect;
    img1(r1).copyTo(img1Rect);

    Mat img2Rect = Mat::zeros(r2.height, r2.width, img1Rect.type());

    applyAffineTransform(img2Rect, img1Rect, t1Rect, t2Rect);

    multiply(img2Rect,mask, img2Rect);
    multiply(img2(r2), Scalar(1.0,1.0,1.0) - mask, img2(r2));
    img2(r2) = img2(r2) + img2Rect;


}

std::vector <cv::Point2f> get_points(const dlib::full_object_detection& d)
{
    std::vector <cv::Point2f> points;
    for (int i = 0; i < 68; ++i)
    {
        points.push_back(cv::Point2f(d.part(i).x(), d.part(i).y()));
    }

    return points;
}

void save_points(const dlib::full_object_detection& d)
{
    std::vector <cv::Point2f> points = get_points(d);
    std::fstream outputFile;
    outputFile.open( "spook.txt", std::ios::out );
    for( size_t ii = 0; ii < points.size( ); ++ii )
     outputFile << points[ii].x << " " << points[ii].y <<std::endl;
    outputFile.close( );
}

void render_face (cv::Mat &img, const dlib::full_object_detection& d)
{
    DLIB_CASSERT
    (
     d.num_parts() == 68,
     "\n\t Invalid inputs were given to this function. "
     << "\n\t d.num_parts():  " << d.num_parts()
     );

    //save_points(d);

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

int capture(cv::VideoCapture cap, cv::Mat spook,  std::vector<Point2f> spookpoints, frontal_face_detector detector, shape_predictor pose_model){
	#define FACE_DOWNSAMPLE_RATIO 1

    std::vector<dlib::rectangle> faces;
    cv::Mat im;
    cv::Mat im_small;

    // Grab and process frames until the main window is closed by the user.
    	cout << "Grabbing a frame..." << endl;
    	cap.set(CV_CAP_PROP_FRAME_WIDTH,1280);   // width pixels
    	cap.set(CV_CAP_PROP_FRAME_HEIGHT,960);   // height pixels
    	if(!cap.isOpened()){   // connect to the camera
    	         cout << "Failed to connect to the camera." << endl;
    	         return 1;
        }
        // Grab a frame
        cap >> im;
        //im=spook;

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
            return capture(cap, spook, spookpoints, detector, pose_model);
        }
        full_object_detection shape;
        for (unsigned long i = 0; i < faces.size(); ++i)
        {
        	// Resize obtained rectangle for full resolution image.
        	     dlib::rectangle r(
        	                   (long)(faces[i].left() * FACE_DOWNSAMPLE_RATIO),
        	                   (long)(faces[i].top() * FACE_DOWNSAMPLE_RATIO),
        	                   (long)(faces[i].right() * FACE_DOWNSAMPLE_RATIO),
        	                   (long)(faces[i].bottom() * FACE_DOWNSAMPLE_RATIO)
        	                );

        	// Landmark detection on full sized image
        	shape = pose_model(cimg, r);
            //shapes.push_back(pose_model(cimg, faces[i]));
            shapes.push_back(shape);

        }

        Mat img1 = im;
        Mat img2 = spook;
        Mat img1Warped = img2.clone();

        //Read points
        std::vector<Point2f> points1, points2;
        points1 = get_points(shape);
        points2 = spookpoints;

        //convert Mat to float data type
        img1.convertTo(img1, CV_32F);
        img1Warped.convertTo(img1Warped, CV_32F);

        // Find convex hull
        std::vector<Point2f> hull1;
        std::vector<Point2f> hull2;
        std::vector<int> hullIndex;

        convexHull(points2, hullIndex, false, false);

        for(int i = 0; i < hullIndex.size(); i++)
        {
            hull1.push_back(points1[hullIndex[i]]);
            hull2.push_back(points2[hullIndex[i]]);
        }

        // Find delaunay triangulation for points on the convex hull
        std::vector< std::vector<int> > dt;
        Rect rect(0, 0, img1Warped.cols, img1Warped.rows);
        calculateDelaunayTriangles(rect, hull2, dt);

        // Apply affine transformation to Delaunay triangles
        for(size_t i = 0; i < dt.size(); i++)
        {
           std::vector<Point2f> t1, t2;
           // Get points for img1, img2 corresponding to the triangles
           for(size_t j = 0; j < 3; j++)
           {
        	  t1.push_back(hull1[dt[i][j]]);
        	  t2.push_back(hull2[dt[i][j]]);
        	}
                warpTriangle(img1, img1Warped, t1, t2);
       	}

        // Calculate mask
        std::vector<Point> hull8U;
        for(int i = 0; i < hull2.size(); i++)
        {
            Point pt(hull2[i].x, hull2[i].y);
            hull8U.push_back(pt);
        }

        Mat mask = Mat::zeros(img2.rows, img2.cols, img2.depth());
        fillConvexPoly(mask,&hull8U[0], hull8U.size(), Scalar(255,255,255));

        // Clone seamlessly.
        Rect r = boundingRect(hull2);
        Point center = (r.tl() + r.br()) / 2;

        Mat output;
        img1Warped.convertTo(img1Warped, CV_8UC3);
        cv::seamlessClone(img1Warped,img2, mask, center, output, NORMAL_CLONE);

        imshow("Face Swapped", output);
        waitKey(0);
        destroyAllWindows();

        /*
        cout << "Displaying on screen..." << endl;
        // Custom Face Render
        render_face(im, shape);
        // Display it all on the screen
        //win.clear_overlay();
        //win.set_image(cimg);
        //win.add_overlay(render_face_detections(shapes));
        // We can also extract copies of each face that are cropped, rotated upright,
        // and scaled to a standard size as shown here:
        dlib::array<array2d<rgb_pixel> > face_chips;
        extract_image_chips(cimg, get_face_chip_details(shapes), face_chips);
        image_window win_faces;
        win_faces.set_image(tile_images(face_chips));

        while(!win_faces.is_closed()){} */
        return 0;
}

int main(int argc, char** argv)
{
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
        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        cout << "Reading in shape predictor..." << endl;
        deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
        cout << "Done reading in shape predictor ..." << endl;

        cout << "Reading in spook..." << endl;
        cv::Mat spook = cv::imread("spook.png", CV_LOAD_IMAGE_COLOR);
        std::vector<Point2f> spookpoints = readPoints("spook.txt");

        return capture(cap, spook, spookpoints, detector, pose_model);

        // Release the cam
        cap.release();
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


