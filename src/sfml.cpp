#include <atomic>
#include <iostream>
#include <string>
#include <thread>
#include <csignal>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>

using namespace sf;
using namespace cv;
using namespace dlib;
using namespace std;

#define FACE_DOWNSAMPLE_RATIO 4

std::atomic_int initct(0),initmt(0),stopping(0);
cv::Mat frameBGR;
cv::Mat frameRGB;
std::mutex m;

void draw_polyline(cv::Mat &img, const dlib::full_object_detection& d, const int start, const int end, bool isClosed = false)
{
    std::vector <cv::Point> points;
    for (int i = start; i <= end; ++i)
    {
        points.push_back(cv::Point(d.part(i).x(), d.part(i).y()));
        cv::circle(img, cv::Point(d.part(i).x(), d.part(i).y()), 2, cv::Scalar(255,0,0), 2, 16, 0);
    }
    cv::polylines(img, points, isClosed, cv::Scalar(255,0,0), 2, 16);
}

void renderFace(cv::Mat &img, const dlib::full_object_detection& d)
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

void captureThread(int devnum){

	cout << "Entering captureThread." << endl;
	cv::VideoCapture cap(devnum); // open the video file for reading
	cv::Size size(1600, 900);
	cv::Mat capBGR;

    if(!cap.isOpened())
	{
	    return;
	}
	cap.set(CV_CAP_PROP_FRAME_WIDTH,1920);   // width pixels
	cap.set(CV_CAP_PROP_FRAME_HEIGHT,1080);   // height pixels
	cap.set(CV_CAP_PROP_FPS,30);
	std::cout << "Size " << cap.get(CV_CAP_PROP_FRAME_WIDTH) << " x "
	    << cap.get(CV_CAP_PROP_FRAME_HEIGHT) << " at " << cap.get(CV_CAP_PROP_FPS)
		<< " fps." << endl;

	while(!stopping.load())
    {
		cap >> capBGR;
		cv::flip(capBGR, capBGR, 1);
        cv::resize(capBGR, capBGR, size);
        if(capBGR.empty())
        {
            break;
        }
        std::unique_lock<std::mutex> l(m);
        frameBGR = capBGR.clone();
        initct.store(1);
    }
	std::cout << "Capturethread ending! " << std::endl;
}

void renderingThread(sf::RenderWindow *window)
{

	sf::Image image;
	sf::Texture texture;
	sf::Sprite sprite;

    // the rendering loop
    while (window->isOpen())
    {

    	image.create(frameRGB.cols, frameRGB.rows, frameRGB.ptr());

        if (!texture.loadFromImage(image))
        {
            break;
        }

        sprite.setTexture(texture);

        window->draw(sprite);
        window->display();

		sf::Event event;
		/* Some workload may be here */
        while (window->pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window->close();
            if (event.type == sf::Event::KeyPressed)
            {
                if (event.key.code == sf::Keyboard::Escape)
                {
                    std::cout << "the escape key was pressed" << std::endl;
                    window->close();
                }
            }
        }

    }
}

void modelThread(){
	// Load face detection and pose estimation models.
	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pose_model;
	std::vector<dlib::rectangle> faces;
	cv::Mat frameBGRsmall;

	cout << "Reading in shape predictor..." << endl;
	deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
	cout << "Done reading in shape predictor..." << endl;

	while(!stopping.load())
	{
	  cv::resize(frameBGR, frameBGRsmall, cv::Size(), 1.0/FACE_DOWNSAMPLE_RATIO, 1.0/FACE_DOWNSAMPLE_RATIO);
      cv_image<bgr_pixel> cimg(frameBGRsmall);

      //mframeRGB.unlock();

      // Detect faces
	  faces = detector(cimg);
	  // Find the pose of each face.
	  std::vector<full_object_detection> shapes;
	  if (faces.size() == 0)
	  {
	   	//cout << "No faces detected." << endl;
	   	cv::cvtColor(frameBGR, frameRGB, cv::COLOR_BGR2RGBA);
	   	continue;
	  }
      full_object_detection shape;
      //cout << "There were " << faces.size() << " faces detected." << endl;
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

      std::unique_lock<std::mutex> l(m);
      cv::cvtColor(frameBGR, frameRGB, cv::COLOR_BGR2RGBA);
      for (unsigned long i = 0; i < shapes.size(); ++i){
       	 renderFace(frameRGB, shapes[i]);
       }

    initmt.store(1);
   }
}

int main(int argc, char** argv){

	if (argc != 2)
	{
	  cout << "Call this program with a single digit number to indicate the /dev/video(x) input to use." << endl;
	  return 0;
    }

	if (!(isdigit(argv[1][0])))
	{
      cout << "Parameter " << argv[1][0] << " is not a number." << endl;
      return 0;
	}

    std::thread ct = std::thread(captureThread, atoi(argv[1]));
	std::thread mt = std::thread(modelThread);

	while(!initct.load());
	while(!initmt.load());

	sf::RenderWindow window(sf::VideoMode(1600, 900), "RenderWindow",sf::Style::Fullscreen);
	//sf::RenderWindow window(sf::VideoMode(1200, 600), "RenderWindow");
    window.setMouseCursorVisible(false);
	window.setActive(false);

	sf::Music music;
	if (!music.openFromFile("BennyHill.ogg"))
	    return -1; // error
	music.play();
	music.setLoop(true);

	std::thread rt = std::thread(&renderingThread,&window);

	while (window.isOpen())
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}

	music.stop();
	stopping.store(1);

	ct.join();
	mt.join();
	rt.join();

}
