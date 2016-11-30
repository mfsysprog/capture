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
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

#include "FaceSwapper.h"

using namespace sf;
using namespace cv;
using namespace dlib;
using namespace std;

#define FACE_DOWNSAMPLE_RATIO 4

std::atomic_int initct(0),initmt(0),stopping(0);
cv::Mat frameBGR;
cv::Mat frameRGB;
std::mutex m;
shape_predictor pose_model;
int source_hist_int[3][256];
int target_hist_int[3][256];
float source_histogram[3][256];
float target_histogram[3][256];

class LaplacianBlending {
private:
    Mat_<Vec3f> left;
    Mat_<Vec3f> right;
    Mat_<float> blendMask;

    std::vector<Mat_<Vec3f> > leftLapPyr,rightLapPyr,resultLapPyr;
    Mat leftSmallestLevel, rightSmallestLevel, resultSmallestLevel;
    std::vector<Mat_<Vec3f> > maskGaussianPyramid; //masks are 3-channels for easier multiplication with RGB

    int levels;


    void buildPyramids() {
        buildLaplacianPyramid(left,leftLapPyr,leftSmallestLevel);
        buildLaplacianPyramid(right,rightLapPyr,rightSmallestLevel);
        buildGaussianPyramid();
    }

    void buildGaussianPyramid() {
        assert(leftLapPyr.size()>0);

        maskGaussianPyramid.clear();
        Mat currentImg;
        cvtColor(blendMask, currentImg, CV_GRAY2BGR);
        maskGaussianPyramid.push_back(currentImg); //highest level

        currentImg = blendMask;
        for (unsigned int l=1; l<levels+1; l++) {
            Mat _down;
            if (leftLapPyr.size() > l) {
                pyrDown(currentImg, _down, leftLapPyr[l].size());
            } else {
                pyrDown(currentImg, _down, leftSmallestLevel.size()); //smallest level
            }

            Mat down;
            cvtColor(_down, down, CV_GRAY2BGR);
            maskGaussianPyramid.push_back(down);
            currentImg = _down;
        }
    }

    void buildLaplacianPyramid(const Mat& img, std::vector<Mat_<Vec3f> >& lapPyr, Mat& smallestLevel) {
        lapPyr.clear();
        Mat currentImg = img;
        for (int l=0; l<levels; l++) {
            Mat down,up;
            pyrDown(currentImg, down);
            pyrUp(down, up, currentImg.size());
            Mat lap = currentImg - up;
            lapPyr.push_back(lap);
            currentImg = down;
        }
        currentImg.copyTo(smallestLevel);
    }

    Mat_<Vec3f> reconstructImgFromLapPyramid() {
        Mat currentImg = resultSmallestLevel;
        for (int l=levels-1; l>=0; l--) {
            Mat up;

            pyrUp(currentImg, up, resultLapPyr[l].size());
            currentImg = up + resultLapPyr[l];
        }
        return currentImg;
    }

    void blendLapPyrs() {
        resultSmallestLevel = leftSmallestLevel.mul(maskGaussianPyramid.back()) +
                                    rightSmallestLevel.mul(Scalar(1.0,1.0,1.0) - maskGaussianPyramid.back());
        for (int l=0; l<levels; l++) {
            Mat A = leftLapPyr[l].mul(maskGaussianPyramid[l]);
            Mat antiMask = Scalar(1.0,1.0,1.0) - maskGaussianPyramid[l];
            Mat B = rightLapPyr[l].mul(antiMask);
            Mat_<Vec3f> blendedLevel = A + B;

            resultLapPyr.push_back(blendedLevel);
        }
    }

public:
    LaplacianBlending(const Mat_<Vec3f>& _left, const Mat_<Vec3f>& _right, const Mat_<float>& _blendMask, int _levels):
    left(_left),right(_right),blendMask(_blendMask),levels(_levels)
    {
        assert(_left.size() == _right.size());
        assert(_left.size() == _blendMask.size());
        buildPyramids();
        blendLapPyrs();
    };

    Mat_<Vec3f> blend() {
        return reconstructImgFromLapPyramid();
    }
};

Mat_<Vec3f> LaplacianBlend(const Mat_<Vec3f>& l, const Mat_<Vec3f>& r, const Mat_<float>& m) {
    LaplacianBlending lb(l,r,m,4);
    return lb.blend();
}

// Apply affine transform calculated using srcTri and dstTri to src
void applyAffineTransform(Mat &warpImage, Mat &src, std::vector<Point2f> &srcTri, std::vector<Point2f> &dstTri)
{
    // Given a pair of triangles, find the affine transform.
    Mat warpMat = getAffineTransform( srcTri, dstTri );

    // Apply the Affine Transform just found to the src image
    warpAffine( src, warpImage, warpMat, warpImage.size(), INTER_LINEAR, BORDER_REFLECT_101);
}

// Warps and alpha blends triangular regions from img1 and img2 to img
void warpTriangle(Mat &img1, Mat &img2, std::vector<Point2f> &t1, std::vector<Point2f> &t2)
{

    cv::Rect r1 = boundingRect(t1);
    cv::Rect r2 = boundingRect(t2);

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


void draw_polyline(cv::Mat &img, const dlib::full_object_detection& d, const int start, const int end, bool isClosed = false)
{
    std::vector <cv::Point> points;
    for (int i = start; i <= end; ++i)
    {
        points.push_back(cv::Point(d.part(i).x(), d.part(i).y()));
        cv::circle(img, cv::Point(d.part(i).x(), d.part(i).y()), 2, cv::Scalar(0,255,0), 8, 16, 0);
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

std::vector <cv::Point2f> get_points(const dlib::full_object_detection& d)
{
    std::vector <cv::Point2f> points;
    for (int i = 0; i < 68; ++i)
    {
    	points.push_back(cv::Point2f(d.part(i).x(), d.part(i).y()));
    }

    return points;
}

// Calculate Delaunay triangles for set of points
// Returns the vector of indices of 3 points for each triangle
static void calculateDelaunayTriangles(cv::Rect rect, std::vector<Point2f> &points, std::vector< std::vector<int> > &delaunayTri){

	// Create an instance of Subdiv2D
    Subdiv2D subdiv(rect);

	// Insert points into subdiv
    for( std::vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
    	if(rect.contains(Point2f(it.base()->x, it.base()->y)))
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
		pt[2] = Point2f(t[4], t[5]);

		if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])){
			for(int j = 0; j < 3; j++)
				for(size_t k = 0; k < points.size(); k++)
         			if(abs(pt[j].x - points[k].x) < 1.0 && abs(pt[j].y - points[k].y) < 1.0)
     					ind[j] = k;

			delaunayTri.push_back(ind);
		}
	}
}

void captureThread(int devnum){

	cout << "Entering captureThread." << endl;
	cv::VideoCapture cap(devnum); // open the video file for reading

	//cv::Size size(1600, 900);
    cv::Size size(800, 600);
	cv::Mat capBGROrig;
	cv::Mat capBGR;

    if(!cap.isOpened())
	{
    	initct.store(2);
        return;
	}
	//cap.set(CV_CAP_PROP_FRAME_WIDTH,1920);   // width pixels
	//cap.set(CV_CAP_PROP_FRAME_HEIGHT,1080);   // height pixels
	cap.set(CV_CAP_PROP_FRAME_WIDTH,800);   // width pixels
	cap.set(CV_CAP_PROP_FRAME_HEIGHT,600);   // height pixels
	cap.set(CV_CAP_PROP_FPS,30);
	std::cout << "Size " << cap.get(CV_CAP_PROP_FRAME_WIDTH) << " x "
	    << cap.get(CV_CAP_PROP_FRAME_HEIGHT) << " at " << cap.get(CV_CAP_PROP_FPS)
		<< " fps." << endl;

	while(!stopping.load())
    {
		cap >> capBGROrig;
		cv::flip(capBGROrig, capBGROrig, 1);
        cv::resize(capBGROrig, capBGR, size);
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
    	{
    	  std::unique_lock<std::mutex> l(m);
    	  image.create(frameRGB.cols, frameRGB.rows, frameRGB.ptr());
    	}

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

  while(!stopping.load())
  {
    try
	{

	  std::vector<dlib::rectangle> faces;
      cv::Mat modelBGR;
      cv::Mat modelBGRWarped;
      cv::Mat modelBGRsmall;

	  {
    	  std::unique_lock<std::mutex> l(m);
		  modelBGR = frameBGR.clone();
		  modelBGRWarped = frameBGR.clone();
	  }

	  cv::resize(modelBGR, modelBGRsmall, cv::Size(), 1.0/FACE_DOWNSAMPLE_RATIO, 1.0/FACE_DOWNSAMPLE_RATIO);
	  cv_image<bgr_pixel> cimg(modelBGRsmall);
	  cv_image<bgr_pixel> img(modelBGR);

      // Detect faces
	  faces = detector(cimg);
	  if (faces.size() == 0)
	  {
	   	//cout << "No faces detected." << endl;
		std::unique_lock<std::mutex> l(m);
	   	cv::cvtColor(modelBGR, frameRGB, cv::COLOR_BGR2RGBA);
	   	initmt.store(1);
	   	continue;
	  }

      std::vector<full_object_detection> shapes;
      std::vector<std::vector<Point2f>> points;
      std::vector<std::vector<Point2f>> hulls;
      std::vector<std::vector< std::vector<int> >> dts;

      //cout << "There were " << faces.size() << " faces detected." << endl;

      //convert to float
      //cv::Mat imgOrig;
      //frameRGB.convertTo(imgOrig, CV_32F);
      //cv::Mat imgNew = imgOrig.clone();

      for (unsigned long i = 0; i < faces.size(); ++i)
      {
    	full_object_detection shape;
    	std::vector<Point2f> point;
    	std::vector<Point2f> hull;
    	std::vector<int> hullIndex;
    	std::vector<std::vector<int>> dt;

       	// Resize obtained rectangle for full resolution image.
        dlib::rectangle r(
         (long)(faces[i].left() * FACE_DOWNSAMPLE_RATIO),
         (long)(faces[i].top() * FACE_DOWNSAMPLE_RATIO),
         (long)(faces[i].right() * FACE_DOWNSAMPLE_RATIO),
         (long)(faces[i].bottom() * FACE_DOWNSAMPLE_RATIO)
        );
      	// Landmark detection on full sized image
      	shape = pose_model(img, r);
      	point = get_points(shape);

      	/*
      	//calculate forehead left
    	point.push_back(cv::Point2f((point[5].x+point[18].x)/2,point[18].y-0.25*(point[5].y-point[18].y)));
        //calculate forehead middle
        point.push_back(cv::Point2f((point[8].x+point[27].x)/2,point[27].y-0.25*(point[8].y-point[27].y)));
        //calculate forehead right
        point.push_back(cv::Point2f((point[11].x+point[25].x)/2,point[25].y-0.25*(point[11].y-point[25].y)));
		*/

      	//find convex hull
        convexHull(point, hullIndex, false, false);

        for(int i = 0; i < (int)hullIndex.size(); i++)
        {
            hull.push_back(point[hullIndex[i]]);
        }

        cv::Rect rect(0, 0, modelBGR.cols, modelBGR.rows);

        calculateDelaunayTriangles(rect, point, dt);

        //save
        hulls.push_back(hull);
        points.push_back(point);
      	shapes.push_back(shape);
      	dts.push_back(dt);
      }

      /*

      for (unsigned long i = 0; i < shapes.size(); i++){
               	 renderFace(modelBGR, shapes[i]);
      }

      std::vector<cv::Point> hullpoint;

      for (unsigned long i = 0; i < hulls.size(); i++){
             hullpoint.clear();
             hullpoint.resize(hulls[i].size());
       	     //transform point2f to point2i
    	     std::transform(hulls[i].begin(),hulls[i].end(),hullpoint.begin(),[](Point2f point2f){return (Point) point2f;});
    	     cv::polylines(modelBGR, hullpoint, true, cv::Scalar(255,0,0), 4, CV_AA);
      }

      Scalar delaunay_color(255,255,255);

      for (unsigned long i = 0; i < dts.size(); i++)
      {
    	  for (unsigned long k = 0; k < dts[i].size(); k++)
              {
                  cv::line(modelBGR,
                		   points[i][dts[i][k][0]],
						   points[i][dts[i][k][1]],
                		   delaunay_color, 1, CV_AA, 0);
                  cv::line(modelBGR,
                		   points[i][dts[i][k][1]],
						   points[i][dts[i][k][2]],
                		   delaunay_color, 1, CV_AA, 0);
                  cv::line(modelBGR,
                		   points[i][dts[i][k][2]],
						   points[i][dts[i][k][0]],
                		   delaunay_color, 1, CV_AA, 0);

              }
      }
	  */
      //convert Mat to float data type
      modelBGR.convertTo(modelBGR, CV_32F);
      modelBGRWarped.convertTo(modelBGRWarped, CV_32F);

      // Apply affine transformation to Delaunay triangles
      if (dts.size() > 1)
      {
    	for(unsigned int i = 0; i < dts.size(); i++)
    	{
      	  for(unsigned int k = 0; k < dts[i].size(); k++)
            {
              std::vector<Point2f> t1, t2;
              // Get points for img1, img2 corresponding to the triangles
      		  for(size_t j = 0; j < 3; j++)
                {
      			  t1.push_back(points[i][dts[i][k][j]]);
       		      t2.push_back(points[((i+1) % dts.size())][dts[i][k][j]]);
      		    }

              warpTriangle(modelBGR, modelBGRWarped, t1, t2);
            }
      	}
      }
      else
      {
    	  modelBGRWarped = modelBGR.clone();
      }

      modelBGRWarped.convertTo(modelBGRWarped, CV_8UC3);
      modelBGR.convertTo(modelBGR, CV_8UC3);
      FaceSwapper face_swapper;

      if (hulls.size() > 1)
      // Calculate mask
      for(unsigned int i = 0; i < hulls.size(); i++)
      {
    	  std::vector<Point> hull8U;
    	  hull8U.clear();
          cv::Rect r = boundingRect(hulls[i]);

          for(unsigned int k = 0; k < hulls[i].size(); k++)
          {
        	  Point pt(hulls[i][k].x, hulls[i][k].y);
              hull8U.push_back(pt);
          }

          Mat mask = Mat::zeros(modelBGR.rows, modelBGR.cols, modelBGR.depth());

          fillConvexPoly(mask,&hull8U[0], hull8U.size(), Scalar(255,0,0));
          Mat output;
          face_swapper.specifiyHistogram(modelBGR(r), modelBGRWarped(r), mask(r));

          mask.convertTo(mask,CV_32F,1.0/255.0);
          Mat_<Vec3f> left; modelBGRWarped.convertTo(left,CV_32F,1.0/255.0);
          Mat_<Vec3f> right; modelBGR.convertTo(right,CV_32F,1.0/255.0);
          Mat_<Vec3f> blend = LaplacianBlend(left(r), right(r), mask(r));
          modelBGRWarped.convertTo(modelBGRWarped, CV_32F, 1.0/255.0);
          blend.copyTo(modelBGRWarped(r));
          modelBGRWarped.convertTo(modelBGRWarped,CV_8UC3,255);

          /*
          seamlessClone(modelBGRWarped(r),modelBGR(r), mask(r),
        		  Point(modelBGR(r).cols / 2, modelBGR(r).rows / 2), output, NORMAL_CLONE);
          output.copyTo(modelBGRWarped(r));
          */
      }

      std::unique_lock<std::mutex> l(m);
      	  cv::cvtColor(modelBGRWarped, frameRGB, cv::COLOR_BGR2RGBA);

      initmt.store(1);
	}

	catch(const std::exception& e)
	{
	   cout << "Exception : " << e.what() << endl;
	}


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

	cout << "Reading in shape predictor..." << endl;
	deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
	cout << "Done reading in shape predictor..." << endl;

    std::thread ct = std::thread(captureThread, atoi(argv[1]));

	while(!initct.load());
    if (initct.load() == 2)
			{
			   cout << "Unable to open webcam /dev/video" << argv[1][0] << endl;
               ct.join();
			   return -1;
			}

    std::thread mt = std::thread(modelThread);

	while(!initmt.load());

	//sf::RenderWindow window(sf::VideoMode(1600, 900), "RenderWindow",sf::Style::Fullscreen);
	sf::RenderWindow window(sf::VideoMode(640, 480), "RenderWindow");
    window.setMouseCursorVisible(false);
	window.setActive(false);

	sf::Music music;
	if (!music.openFromFile("BennyHill.ogg"))
	{
	   cout << "Unable to load music. " << endl;
	}
	else
	{
       music.play();
	   music.setLoop(true);
	}

	std::thread rt = std::thread(renderingThread,&window);

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
