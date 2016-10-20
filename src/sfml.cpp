#include <SFML/Graphics.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <iostream>

using namespace sf;
using namespace cv;
using namespace std;

int main(int argc, char** argv){

cv::VideoCapture cap(0); // open the video file for reading
if(!cap.isOpened())
{
    return 0;
}
cap.set(CV_CAP_PROP_FRAME_WIDTH,1920);   // width pixels
cap.set(CV_CAP_PROP_FRAME_HEIGHT,1080);   // height pixels

sf::RenderWindow window(sf::VideoMode(1600, 900), "RenderWindow",sf::Style::Fullscreen);
window.setMouseCursorVisible(false);

cv::Mat frameRGB, frameRGBA;
sf::Image image;
sf::Texture texture;
sf::Event event;
sf::Sprite sprite;

cv::Size size(1600, 900);

while (window.isOpen())
{
    cap >> frameRGB;
    cv::flip(frameRGB, frameRGB, 1);
    cv::resize(frameRGB, frameRGB, size);
    if(frameRGB.empty())
    {
        break;
    }
    //std::cout << "Size " << frameRGB.cols << " x " << frameRGB.rows << endl;

    cv::cvtColor(frameRGB, frameRGBA, cv::COLOR_BGR2RGBA);

    image.create(frameRGBA.cols, frameRGBA.rows, frameRGBA.ptr());

    if (!texture.loadFromImage(image))
    {
        break;
    }

    sprite.setTexture(texture);

    while (window.pollEvent(event))
    {
        if (event.type == sf::Event::Closed)
            window.close();
        if (event.type == sf::Event::KeyPressed)
        {
            if (event.key.code == sf::Keyboard::Escape)
            {
                std::cout << "the escape key was pressed" << std::endl;
                window.close();
            }
        }
    }

    window.draw(sprite);
    window.display();

}

}
