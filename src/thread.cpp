#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>

using namespace sf;
using namespace std;


#define FACE_DOWNSAMPLE_RATIO 4

void captureThread(){
	cout << "Entering captureThread." << endl;
	cout << "Capturethread ending! " << endl;
}

int main(int argc, char** argv){

  	std::thread ct = std::thread(captureThread);

	//sf::RenderWindow window(sf::VideoMode(1600, 900), "RenderWindow",sf::Style::Fullscreen);
	sf::RenderWindow window(sf::VideoMode(640, 480), "RenderWindow");
    window.setMouseCursorVisible(false);
	//window.setActive(false);

    window.setFramerateLimit(30);
    window.setVerticalSyncEnabled(true);

    sf::Music music;
	music.openFromFile("BennyHill.ogg");
    music.play();
	music.setLoop(true);

    // the rendering loop
    while (window.isOpen())
    {
    	sf::Event event;
		/* Some workload may be here */
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::KeyPressed)
            {
                if (event.key.code == sf::Keyboard::Escape)
                {
                    std::cout << "the escape key was pressed" << std::endl;
                    window.close();
                }
            }
        }
        window.display();
    }

	music.stop();

	ct.join();

}
