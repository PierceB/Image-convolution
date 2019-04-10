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
#define TILE_WIDTH 16
#define FILTERDIM 3                     //CHANGE THIS WHEN USING DIFFERENT MASK SIZE
#define PW (TILE_WIDTH + FILTERDIM - 1 )

texture<float, 2, cudaReadModeElementType> tex;

const char *imageFilename = "lena_bw.pgm";

const char *sampleName = "HPCAssignment2.cu";


//Texture memory kernel=========================================================
__global__ void GPUTextureConv(float* doutput, float* filter, int width, int height, int filterDim){

		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;           //calculate x coord
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;                       //calculate y coord

    float u = (float)x +0.5f;                                                //convert x and y coordinate, include pixel width to get center
    float v = (float)y +0.5f;

		int offset = ((filterDim-1)/2);                              //calculate offset from the filter, so if its 3x3 it will be 1
		int i = y*width +x ;                                       //calculate index in the 1D array for output data
		float sum=0.0;                                           //initlize sum

		for(int k=0;k<filterDim;k++){
			for(int l = 0; l<filterDim;l++){
			if((u-offset+l >= 0 )&&(u-offset+l < width ) && (v-offset+k >=0) && (v-offset+k < height))  //make sure point is within image boundary
			sum+=	tex2D(tex,u-offset+l,v-offset+k)*filter[l+k*filterDim];     //fetch pixel data from texture memory and multiply by corresponding filter value
			}
		}

		if(sum<0)                    //normalize
			sum=0;
		if(sum>1)
			sum=1 ;
			doutput[i] = sum;
	}


//==============================================================================


//SHARED MEMORY TILING IMPLEMENTATION===========================================
//do later
__global__ void GPUSharedConv(float* ddata, float* doutput, float* filter,int blockWidth, int imageWidth, int imageHeight, int filterDim){





}


__constant__ float dconstantFilter[FILTERDIM*FILTERDIM];            //define array for filter in constant memory
//CONSTANT MEMORY FILTER IMPLEMENTATION=========================================
__global__ void GPUConstantConv(float* ddata,const float *__restrict__ kernel, float* doutput,int imageWidth, int imageHeight, int filterDim){

	int k,l;                          //counting variables
	float sum=0.0;                          //temp sum
	int offset = ((filterDim-1)/2);     //bounds for inner loop

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;              //find x dimension
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;		           //find y dimension

	int i = y*imageWidth+ x ;                                          //find unique index for each gpu

	for(k=0; k<filterDim; k++){                                       //calculate CONVOLUTION
		for(l=0; l<filterDim ; l++){
			if((i+l-offset+(k-offset)*imageWidth >= 0) && (i+l-offset+(k-offset)*imageWidth< imageWidth*imageHeight) && (i%imageWidth + l - offset >= 0) && (i%imageWidth + l - offset < imageWidth) && (i%imageHeight + k - offset >=0) && (i%imageHeight +k - offset < imageHeight))                        //COnditions if the filter falls over the image or off. First 2 check the width and last 2 check the heights

				sum+= ddata[i+l-offset+(k-offset)*imageWidth]*dconstantFilter[l+k*filterDim] ;
			}
		}

		if(sum<0)
			sum=0;                                                     //normalise values
		if(sum>1)
			sum=1 ;
			doutput[i] = sum ;                                      //assign output

}

//==============================================================================

//==============================================================================

//NAIVE GPU IMPLEMENTATION OF CONVOLUTION=======================================
__global__ void GPUNaiveConv(float* ddata, float* doutput, float* filter, int imageWidth, int imageHeight, int filterDim){

	int k,l;                          //counting variables
	float sum=0.0;                          //temp sum
	int offset = ((filterDim-1)/2);     //bounds for inner loop

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	int i = y*imageWidth+ x ;

	for(k=0; k<filterDim; k++){
		for(l=0; l<filterDim ; l++){
			if((i+l-offset+(k-offset)*imageWidth >= 0) && (i+l-offset+(k-offset)*imageWidth< imageWidth*imageHeight) && (i%imageWidth + l - offset >= 0) && (i%imageWidth + l - offset < imageWidth) && (i%imageHeight + k - offset >=0) && (i%imageHeight +k - offset < imageHeight))                        //COnditions if the filter falls over the image or off. First 2 check the width and last 2 check the heights
				sum+= ddata[i+l-offset+(k-offset)*imageWidth]*filter[l+k*filterDim] ;
		}
	}
	if(sum<0)
		sum=0;
	if(sum>1)
		sum=1 ;

	doutput[i] = sum ;


}
//==============================================================================


//CPU IMPLEMENTATION OF CONVOLUTION=============================================
void CPUConv(float* hdata, float* houtput, float* filter, int imageWidth, int imageHeight, int filterDim){

	int i,k,l;                          //counting variables
	float sum;                          //temp sum
	int offset = floor(filterDim/2);     //bounds for inner loop
//	printf("%d \n",imageWidth*imageHeight);
	for(i=0; i< imageWidth*imageHeight; i++){
		sum=0.0;
		for(k=0; k<filterDim; k++){
			for(l=0; l<filterDim ; l++){
				if((i+l-offset+(k-offset)*imageWidth >= 0) && (i+l-offset+(k-offset)*imageWidth< imageWidth*imageHeight) && (i%imageWidth + l - offset >= 0) && (i%imageWidth + l - offset < imageWidth) && (i%imageHeight + k - offset >=0) && (i%imageHeight +k - offset < imageHeight))                        //COnditions if the filter falls over the image or off. First 2 check the width and last 2 check the heights
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

//==============================================================================

int main(int argc, char **argv){

	int devID = findCudaDevice(argc, (const char **) argv);
    // load image from disk
    	float *hData = NULL;                                                 //To store the image
    	unsigned int width, height;
    	char *imagePath = sdkFindFilePath(imageFilename, argv[0]);           //find the image path

//DEFINE FILTER HERE============================================================
//	float filter[] ={0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0};                 //returns original image
	 float filter[] ={-1.0,0.0,1.0,-2.0,0.0,2.0,-1.0,0.0,1.0};            //highlights edges
//	float filter[] ={-1.0,-1.0,-1.0,-1.0,9.0,-1.0,-1.0,-1.0,-1.0};      //sharpens image
//	float filter[] ={1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0};   //slightly blurs image

	int filterDim=FILTERDIM;  						//dimensions of the filter, assume its square
//==============================================================================

    	if (imagePath == NULL)
    	{
       	 printf("Unable to source image file: %s\n", imageFilename);
       	 exit(EXIT_FAILURE);
    	}

    	sdkLoadPGM(imagePath, &hData, &width, &height);             //load image into hdata and initialize width and height, hdata is a 1D array
			//printf("%d %d \n", width,height);
    	unsigned int size = width * height * sizeof(float);                 //get total size of image (length of array) in bytes
    	float *hOutputData = (float *) malloc(size);                        //create an array to store the final


//RUN CPU VERSION ==============================================================
		cudaEvent_t cpulaunch_begin, cpulaunch_end;
		cudaEventCreate(&cpulaunch_begin);
		cudaEventCreate(&cpulaunch_end);
// record a CUDA event immediately before and after the kernel launch
		cudaEventRecord(cpulaunch_begin,0);

 	  CPUConv(hData,hOutputData,filter,width,height,filterDim);   //Run cpu version

		cudaEventRecord(cpulaunch_end,0);
 		cudaEventSynchronize(cpulaunch_end);
// measure the time (ms) spent in the kernel
	 float cputime = 0;
	 cudaEventElapsedTime(&cputime, cpulaunch_begin, cpulaunch_end);
	 printf("CPU run time: %fms\n", cputime);
	 sdkSavePGM("Image_CPU_OUT.pgm",hOutputData,width,height);
//==============================================================================

 //Run Naive GPU VERSION========================================================
	 cudaEvent_t nglaunch_begin, nglaunch_end;

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

		const size_t block_size = 64;                                 //initialise block size
  	size_t grid_size = width*height / block_size;                   // calculate gride size

  	// deal with a possible partial final block
 	 	if(width*height % block_size) ++grid_size;


  	cudaEventCreate(&nglaunch_begin);
  	cudaEventCreate(&nglaunch_end);
	// record a CUDA event immediately before and after the kernel launch
  	cudaEventRecord(nglaunch_begin,0);
  // launch the kernel
  	GPUNaiveConv<<<grid_size, block_size>>>(dData,dOutput,dFilter,width,height,filterDim) ;          //Call the kernal
		cudaEventRecord(nglaunch_end,0);
 	 	cudaEventSynchronize(nglaunch_end);
	// measure the time (ms) spent in the kernel
  	float ngtime = 0;
  	cudaEventElapsedTime(&ngtime, nglaunch_begin, nglaunch_end);

  // copy the result back to the host memory space
  	cudaMemcpy(hOutputData, dOutput, size, cudaMemcpyDeviceToHost);
		printf("GPU Naive run time: %fms\n", ngtime);

		sdkSavePGM("Image_NAIVE_OUT.pgm",hOutputData,width,height);               //save the new image as Image_out.pgm
//==============================================================================

//RUN THE CONSTANT MEMORY GPU implementation====================================
	cudaEvent_t cglaunch_begin, cglaunch_end;

	float *dcData = 0;
	float *dcOutput = 0;
	float *dcFilter = 0;

	cudaMalloc((void**)&dcData, size);
	cudaMalloc((void**)&dcOutput, size);
	cudaMalloc((void**)&dcFilter, filterDim*filterDim*sizeof(float));       //assign the space required for the above arrays
	cudaMalloc((void**)&dconstantFilter, filterDim*filterDim*sizeof(float));       //assign the space required for the above arrays

 if(dcOutput == 0 || dcData == 0 || dcFilter == 0)                       //check if the arrays actually initialised properly
	{
		printf("couldn't allocate device memory (constant)\n");
		return 1;
	}

	cudaMemcpy(dcData, hData, size, cudaMemcpyHostToDevice);            //Copy the image to the device
	cudaMemcpy(dcFilter, filter, filterDim*filterDim*sizeof(float), cudaMemcpyHostToDevice);	   //copy the filter to the device
	cudaMemcpyToSymbol(dconstantFilter,filter,sizeof(float)*filterDim*filterDim,0,cudaMemcpyHostToDevice) ;



	size_t cgrid_size = width*height / block_size;                   // calculate gride size

  	// deal with a possible partial final block
 	 if(width*height % block_size) ++cgrid_size;

	 cudaEventCreate(&cglaunch_begin);
	 cudaEventCreate(&cglaunch_end);
	 // record a CUDA event immediately before and after the kernel launch
	 cudaEventRecord(cglaunch_begin,0);
	 // launch the kernel
	 GPUConstantConv<<<cgrid_size, block_size>>>(dcData,dconstantFilter,dcOutput,width,height,filterDim) ;          //Call the kernal
	 cudaEventRecord(cglaunch_end,0);
	 cudaEventSynchronize(cglaunch_end);
	 // measure the time (ms) spent in the kernel
	 		float cgtime =0;
	 		cudaEventElapsedTime(&cgtime, cglaunch_begin, cglaunch_end);

	 // copy the result back to the host memory space
	 cudaMemcpy(hOutputData, dcOutput, size, cudaMemcpyDeviceToHost);
	 printf("GPU Const run time: %fms\n", cgtime);

	 sdkSavePGM("Image_CONST_OUT.pgm",hOutputData,width,height);

//==============================================================================

//TEXTURE MEMORY IMPLEMENTATION ================================================
//float *dData = NULL;
	cudaEvent_t tglaunch_begin, tglaunch_end;
	float *dtFilter = 0;                               //create pointer for filter

	 cudaMalloc((void**)&dtFilter, filterDim*filterDim*sizeof(float)); 				//allocate memory for filter
	 cudaMemcpy(dtFilter, filter, filterDim*filterDim*sizeof(float), cudaMemcpyHostToDevice);	   //copy the filter to the device

	 checkCudaErrors(cudaMalloc((void **) &dData, size));

	 // Allocate array and copy image data
	 cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	 cudaArray *cuArray;
	 checkCudaErrors(cudaMallocArray(&cuArray, &channelDesc, width, height));
	 checkCudaErrors(cudaMemcpyToArray(cuArray,0,0, hData,size, cudaMemcpyHostToDevice));

	 // Set texture parameters
	 tex.addressMode[0] = cudaAddressModeWrap;
	 tex.addressMode[1] = cudaAddressModeWrap;
	 tex.filterMode = cudaFilterModeLinear;
	 tex.normalized = false;    // access with normalized texture coordinates

	 // Bind the array to the texture
	 checkCudaErrors(cudaBindTextureToArray(tex, cuArray, channelDesc));

	 dim3 dimBlock(8, 8, 1);				//initlise block and grid size, 2D
	 dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

	 checkCudaErrors(cudaDeviceSynchronize());       //wait for all threads
	 // Execute the kernel

	 cudaEventCreate(&tglaunch_begin);                   //start timer
	 cudaEventCreate(&tglaunch_end);
	 // record a CUDA event immediately before and after the kernel launch
	 cudaEventRecord(tglaunch_begin,0);
	 GPUTextureConv<<<dimGrid, dimBlock, 0>>>(dData,dtFilter, width, height,filterDim);

	 // Check if kernel execution generated an error
	 getLastCudaError("Kernel execution failed");
	 cudaEventRecord(tglaunch_end,0);
	 cudaEventSynchronize(tglaunch_end);
	 // measure the time (ms) spent in the kernel
	 float tgtime =0;
	 cudaEventElapsedTime(&tgtime, tglaunch_begin, tglaunch_end);


	 // copy result from device to host
	 checkCudaErrors(cudaMemcpy(hOutputData,dData,size,	cudaMemcpyDeviceToHost));

	 printf("GPU Texture run time: %fms\n", tgtime);

	 sdkSavePGM("Image_TEXT_OUT.pgm",hOutputData,width,height);

//==============================================================================


/*//Shared memory implementation of convolution===================================
cudaEvent_t cslaunch_begin, cslaunch_end;

float *dsData = 0;
float *dsOutput = 0;
float *dsFilter = 0;

cudaMalloc((void**)&dsData, size);
cudaMalloc((void**)&dsOutput, size);
cudaMalloc((void**)&dsFilter, filterDim*filterDim*sizeof(float));       //assign the space required for the above arrays


if(dsOutput == 0 || dsData == 0 || dsFilter == 0)                       //check if the arrays actually initialised properly
{
	printf("couldn't allocate device memory (shared)\n");
	return 1;
}

int offset = ((filterDim-1)/2);

dim3 sdimBlock(TILE_WIDTH+2*offset, TILE_WIDTH+2*offset, 1);				//initlise block and grid size, 2D
dim3 sdimGrid(width / dimBlock.x, height / dimBlock.y, 1);


 cudaMemcpy(dsData, hData, size, cudaMemcpyHostToDevice);            //Copy the image to the device
 cudaMemcpy(dsFilter, filter, filterDim*filterDim*sizeof(float), cudaMemcpyHostToDevice);	   //copy the filter to the device

 cudaEventCreate(&cslaunch_begin);
 cudaEventCreate(&cslaunch_end);
 // record a CUDA event immediately before and after the kernel launch
 cudaEventRecord(cslaunch_begin,0);
 // launch the kernel
 GPUSharedConv<<<sdimGrid, sdimBlock, 0>>>(dsOutput,dsData,dsFilter, width, height,filterDim);
 cudaEventRecord(cslaunch_end,0);
 cudaEventSynchronize(cslaunch_end);
 // measure the time (ms) spent in the kernel
		float cstime =0;
		cudaEventElapsedTime(&cstime, cslaunch_begin, cslaunch_end);

 // copy the result back to the host memory space
 cudaMemcpy(hOutputData, dsOutput, size, cudaMemcpyDeviceToHost);
 printf("GPU Shared run time: %fms\n", cstime);

 sdkSavePGM("Image_SHARED_OUT.pgm",hOutputData,width,height);*/

//==============================================================================


//	sdkSavePGM("Image_OUT.pgm",hOutputData,width,height);               //save the new image as Image_out.pgm


}
