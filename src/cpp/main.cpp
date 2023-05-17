#include "default.h"

#include <omp.h>
#include <filesystem>
#include <string>

#include <dirent.h> 
#include <stdio.h> 

int main()
{
    int i = 0;
    int threadID = 0;
    std::string fileName = "";
    std::string path = "/home/cewein/Documents/content-0.15-0.20-0.25/Bee-on-Artifical-Flower/yolov7/runs/detect/exp/labels";

    //#pragma omp parallel for private(i, threadID)
    
    DIR *d;
    d = opendir(path.c_str());
    struct dirent * dir;
    

    if (d) {

        #pragma omp parallel private(threadID) shared(dir, d)
        while ((dir = readdir(d)) != NULL)
        {
            threadID = omp_get_thread_num();

            #pragma omp critical
            if(dir != NULL)
            {
                std::cout << "Thread " << threadID << "\t" << dir->d_name << std::endl;
                dir = readdir(d);
            }      
        }
        closedir(d);
    }


    return 0;
}