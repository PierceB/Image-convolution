#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <./helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#define MAX_EPSILON_ERROR 5e-3f

const char *imageFilename = "lena_bw.pgm";
const char *refFilename   = "ref_rotated.pgm";

const char *sampleName = "HPCAssignment2.cu";


void CPUConv(float* hdata, float* houtput, float* filter, int imageWidth, int imageHeight, int filterDim){

	int i,k,l,count;                          //counting variables
	float sum;                          //temp sum 
	int offset = floor(filterDim/2);     //bounds for inner loop
 	
	for(i=0; i< imageWidth*imageHeight; i++){
		sum=0; 
		for(k=0; k<filterDim; k++){
			for(l=0; l<filterDim ; l++){
				if((i%imageWidth + l - offset > 0) && (i%imageWidth + l - offset < imageWidth) && (i%imageWidth + k - offset >0) && (i%imageWidth + l - offset < imageHeight))                        //COnditions if the filter falls over the image or off. First 2 check the width and last 2 check the heights
					sum+= hdata[i+l-offset+(k-offset)*imageWidth]*filter[l+k*filterDim] ; 
			} 
		}
		houtput[i] = sum ; 
	}

}



int main(int argc, char **argv){

	int devID = findCudaDevice(argc, (const char **) argv);

    // load image from disk
    	float *hData = NULL;                                                 //To store the image
    	unsigned int width, height;                       
    	char *imagePath = sdkFindFilePath(imageFilename, argv[0]);           //find the image path

//DEFINE FILTER HERE====================================
	float Nofilter[] ={0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0};                 //returns original image
	float Edgefilter[] ={-1.0,0.0,1.0,-2.0,0.0,2.0,-1.0,0.0,1.0};            //highlights edges
	float Sharpfilter[] ={-1.0,-1.0,-1.0,-1.0,9.0,-1.0,-1.0,-1.0,-1.0};      //sharpens image
	float Blurfilter[] ={1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0};   //slightly blurs image

	int filterDim=3;  						//dimensions of the filter, assume its square
//======================================================

    	if (imagePath == NULL)
    	{
       	 printf("Unable to source image file: %s\n", imageFilename);
       	 exit(EXIT_FAILURE);
    	}

    	sdkLoadPGM(imagePath, &hData, &width, &height);             //load image into hdata and initialize width and height, hdata is a 1D array

    	unsigned int size = width * height * sizeof(float);                 //get total size of image (length of array)
    	float *hOutputData = (float *) malloc(size);                        //create an array to store the final 

		CPUConv(hData,hOutputData,filter,width,height,filterDim);   //Run cpu version

		for(int i=0;i<width*height;i++)
		printf("%f " , hOuputData[i]);	                         //testing purposes
	

	sdkSavePGM("Image_out.pgm",hOutputData,width,height);               //save the new image as Image_out.pgm
 

}

