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


//NAIVE GPU IMPLEMENTATION OF CONVOLUTION====================================================================
__global__ void GPUNaiveConv(float* ddata, float* doutput, float* filter, int imageWidth, int imageHeight, int filterDim){

//UNTESTED

	int k,l;                          //counting variables
	float sum=0.0;                          //temp sum
	int offset = ((filterDim-1)/2);     //bounds for inner loop

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	int i = y*imageWidth+ x ;

	for(k=0; k<filterDim; k++){
		for(l=0; l<filterDim ; l++){
			if((i%imageWidth + l - offset > 0) && (i%imageWidth + l - offset < imageWidth) && (i%imageWidth + k - offset >0) && (i%imageWidth + l - offset < imageHeight))                        //COnditions if the filter falls over the image or off. First 2 check the width and last 2 check the heights
				sum+= ddata[i+l-offset+(k-offset)*imageWidth]*filter[l+k*filterDim] ;
		}
	}
	if(sum<0)
		sum=0;
	if(sum>1)
		sum=1 ;

	doutput[i] = sum ;


}
//===========================================================================================================


//CPU IMPLEMENTATION OF CONVOLUTION==========================================================================
void CPUConv(float* hdata, float* houtput, float* filter, int imageWidth, int imageHeight, int filterDim){

	int i,k,l;                          //counting variables
	float sum;                          //temp sum
	int offset = floor(filterDim/2);     //bounds for inner loop

	for(i=0; i< imageWidth*imageHeight; i++){
		sum=0.0;
		for(k=0; k<filterDim; k++){
			for(l=0; l<filterDim ; l++){
				if((i+l-offset+(k-offset)*imageWidth > 0) && (i+l-offset+(k-offset)*imageWidth < imageWidth*imageHeight) && (i%imageWidth + l - offset > 0) && (i%imageWidth + l - offset < imageWidth) && (i%imageWidth + k - offset >0) && (i%imageWidth + l - offset < imageHeight))                        //COnditions if the filter falls over the image or off. First 2 check the width and last 2 check the heights

					sum+= hdata[i+l-offset+(k-offset)*imageWidth]*filter[l+k*filterDim] ;
			}
		}

		if(sum<0)
			sum=0;
		if(sum>1)
			sum=1 ;
		houtput[i] = sum ;
	}

}

//============================================================================================================

int main(int argc, char **argv){

	int devID = findCudaDevice(argc, (const char **) argv);
	cudaEvent_t launch_begin, launch_end;            //startTimer
    // load image from disk
    	float *hData = NULL;                                                 //To store the image
    	unsigned int width, height;
    	char *imagePath = sdkFindFilePath(imageFilename, argv[0]);           //find the image path

//DEFINE FILTER HERE====================================
//	float filter[] ={0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0};                 //returns original image
	// float filter[] ={-1.0,0.0,1.0,-2.0,0.0,2.0,-1.0,0.0,1.0};            //highlights edges
	float filter[] ={-1.0,-1.0,-1.0,-1.0,9.0,-1.0,-1.0,-1.0,-1.0};      //sharpens image
//	float filter[] ={1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0};   //slightly blurs image

	int filterDim=3;  						//dimensions of the filter, assume its square
//======================================================

    	if (imagePath == NULL)
    	{
       	 printf("Unable to source image file: %s\n", imageFilename);
       	 exit(EXIT_FAILURE);
    	}

    	sdkLoadPGM(imagePath, &hData, &width, &height);             //load image into hdata and initialize width and height, hdata is a 1D array

    	unsigned int size = width * height * sizeof(float);                 //get total size of image (length of array) in bytes
    	float *hOutputData = (float *) malloc(size);                        //create an array to store the final


//RUN CPU VERSION ======================

cudaEventCreate(&launch_begin);
cudaEventCreate(&launch_end);
// record a CUDA event immediately before and after the kernel launch
cudaEventRecord(launch_begin,0);

CPUConv(hData,hOutputData,filter,width,height,filterDim);   //Run cpu version

		cudaEventRecord(launch_end,0);
 		cudaEventSynchronize(launch_end);
// measure the time (ms) spent in the kernel
	 float time = 0;
	 cudaEventElapsedTime(&time, launch_begin, launch_end);
	 printf("CPU run time: %fms\n", time);
	 sdkSavePGM("Image_CPU_OUT.pgm",hOutputData,width,height);
//======================================
 //Run Naive GPU VERSION=================
	//UNTESTED
	float *dData = 0;
	float *dOutput = 0;
	float *dFilter = 0;                      //create arrays we will need to give to device

	cudaMalloc((void**)&dData, size);
 	cudaMalloc((void**)&dOutput, size);
	cudaMalloc((void**)&dFilter, filterDim*filterDim*sizeof(float));       //assign the space required for the above arrays

	 if(dOutput == 0 || dData == 0 || dFilter == 0)                       //check if the arrays actually initialised properly
	  {
	    printf("couldn't allocate device memory\n");
	    return 1;
	  }

	cudaMemcpy(dData, hData, size, cudaMemcpyHostToDevice);            //Copy the image to the device
	cudaMemcpy(dFilter, filter, filterDim*filterDim*sizeof(float), cudaMemcpyHostToDevice);	   //copy the filter to the device

	const size_t block_size = 256;                                 //initialise block size
  	size_t grid_size = width*height / block_size;                   // calculate gride size

  	// deal with a possible partial final block
 	 if(width*height % block_size) ++grid_size;


  	cudaEventCreate(&launch_begin);
  	cudaEventCreate(&launch_end);
	// record a CUDA event immediately before and after the kernel launch
  	cudaEventRecord(launch_begin,0);
  // launch the kernel
  	GPUNaiveConv<<<grid_size, block_size>>>(dData,dOutput,dFilter,width,height,filterDim) ;          //Call the kernal
	cudaEventRecord(launch_end,0);
 	 cudaEventSynchronize(launch_end);
	// measure the time (ms) spent in the kernel
  //	 float time = 0;
       	cudaEventElapsedTime(&time, launch_begin, launch_end);

  // copy the result back to the host memory space
  	cudaMemcpy(hOutputData, dOutput, size, cudaMemcpyDeviceToHost);
	printf("GPU Naive run time: %fms\n", time);

sdkSavePGM("Image_NAIVE_OUT.pgm",hOutputData,width,height);               //save the new image as Image_out.pgm
//======================================


//	for(int i=0;i<width*height;i++)
//	printf("%f " , hOutputData[i]);	                         //testing purposes


	sdkSavePGM("Image_OUT.pgm",hOutputData,width,height);               //save the new image as Image_out.pgm


}
