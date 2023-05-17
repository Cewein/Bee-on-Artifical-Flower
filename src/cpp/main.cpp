#include <omp.h>
#include <filesystem>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

#include <dirent.h> 
#include <stdio.h> 

int main()
{
    //set up path
    int i = 0;
    int threadID = 0;
    std::string fileName = "";
    std::string path = "/home/cewein/Documents/content-0.15-0.20-0.25/Bee-on-Artifical-Flower/yolov7/runs/detect/exp/labels/";

    //set up directory
    DIR *d;
    d = opendir(path.c_str());
    struct dirent * dir;

    if (d) {
        
        //setup parallelisation
        #pragma omp parallel private(threadID) shared(dir, d)
        while ((dir = readdir(d)) != NULL)
        {
            //get thread number
            threadID = omp_get_thread_num();

            //variable for filestream
            int nbBee = 0;
            std::string name = std::string(dir->d_name);

            std::ifstream infile(path+name);
            std::string line;
    
            //read file, each line is a bee
            while (std::getline(infile, line)) nbBee++;

            //output atomically in the console
            #pragma omp critical
            if(dir != NULL)
            {
                std::cout << "Thread " << threadID << "\t" << name << "\tNb Bee : " << nbBee << std::endl;
            }

        }
        closedir(d);
    }


    return 0;
}