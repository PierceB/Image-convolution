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
const float angle = 0.5f;

texture<float, 2, cudaReadModeElementType> tex;

const char *imageFilename = "lena_bw.pgm";

const char *sampleName = "HPCAssignment2.cu";


//Texture memory kernel======================================================================================
__global__ void GPUTextureConv(float* doutput, float* filter, int width, int height, int filterDim){
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float u = (float)x +0.5f;//- (float)width/2;
    float v = (float)y +0.5f;//- (float)height/2;
	//	float tu,tv;
		int offset = ((filterDim-1)/2);
		int i = y*width +x ;
		float sum=0.0;

		for(int k=0;k<filterDim;k++){
			for(int l = 0; l<filterDim;l++){
			if((u-offset+l >= 0 )&&(u-offset+l < width ) && (v-offset+k >=0) && (v-offset+k < height))
			sum+=	tex2D(tex,u-offset+l,v-offset+k)*filter[l+k*filterDim];
			}
		}

		if(sum<0)
			sum=0;
		if(sum>1)
			sum=1 ;
			doutput[i] = sum;
	}


//================================

//First attempt
__global__ void GPU1TextureConv(float* doutput, float* filter, int imageWidth, int imageHeight, int filterDim){
	int k,l;                          //counting variables
	float sum=0.0;                          //temp sum
	int offset = ((filterDim-1)/2);     //bounds for inner loop
	int tu,tv;
	//int tt ;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;              //find x dimension
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;		           //find y dimension
	float u=(x+0.5f)/ (float) imageWidth;
	float v=(y +0.5f)/ (float) imageHeight ;

	int i = y*imageWidth+ x ;                                          //find unique index for each gpu


	for(k=0; k<filterDim; k++){                                       //calculate CONVOLUTION
		for(l=0; l<filterDim ; l++){
			if((i%imageWidth + l - offset > 0) && (i%imageWidth + l - offset < imageWidth) && (i%imageHeight + k - offset >0) && (i%imageHeight +k - offset < imageHeight))                        //COnditions if the filter falls over the image or off. First 2 check the width and last 2 check the heights
			   tu = u+l-offset;
				 tv = v+k-offset;
				 printf("%f ", tex2D(tex,tu,tv)) ;
			//	 tt = i+l-offset+(k-offset)*imageWidth;
				sum+= tex2D(tex,tu,tv)*filter[l+k*filterDim] ;
			}
		}
		if(sum<0)
			sum=0;                                                     //normalise values
		if(sum>1)
			sum=1 ;

			doutput[i] = sum ;                                      //assign output

}



//===========================================================================================================


//SHARED MEMORY TILING IMPLEMENTATION========================================================================
//do later
/*__global__ void GPUSharedConv(float* ddata, float* doutput, float* filter, int imageWidth, int imageHeight, int filterDim){

		int sWidth = TILE_WIDTH + filterDim -1 ;
		__shared__ float SD_data[sWidth][sWidth] ;


}*/
__constant__ float dconstantFilter[FILTERDIM*FILTERDIM];
//CONSTANT MEMORY FILTER IMPLEMENTATION======================================================================
__global__ void GPUConstantConv(float* ddata,const float *__restrict__ kernel, float* doutput,int imageWidth, int imageHeight, int filterDim){

	int k,l;                          //counting variables
	float sum=0.0;                          //temp sum
	int offset = ((filterDim-1)/2);     //bounds for inner loop

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;              //find x dimension
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;		           //find y dimension

	int i = y*imageWidth+ x ;                                          //find unique index for each gpu

	for(k=0; k<filterDim; k++){                                       //calculate CONVOLUTION
		for(l=0; l<filterDim ; l++){
			if((i+l-offset+(k-offset)*imageWidth > 0) && (i+l-offset+(k-offset)*imageWidth< imageWidth*imageHeight) && (i%imageWidth + l - offset > 0) && (i%imageWidth + l - offset < imageWidth) && (i%imageHeight + k - offset >0) && (i%imageHeight +k - offset < imageHeight))                        //COnditions if the filter falls over the image or off. First 2 check the width and last 2 check the heights

				sum+= ddata[i+l-offset+(k-offset)*imageWidth]*dconstantFilter[l+k*filterDim] ;
			}
		}

		if(sum<0)
			sum=0;                                                     //normalise values
		if(sum>1)
			sum=1 ;
///printf("%f %f  %f ",dconstantFilter[0],dconstantFilter[1],dconstantFilter[2]);
			doutput[i] = sum ;                                      //assign output

}

//===========================================================================================================

//===========================================================================================================

//NAIVE GPU IMPLEMENTATION OF CONVOLUTION====================================================================
__global__ void GPUNaiveConv(float* ddata, float* doutput, float* filter, int imageWidth, int imageHeight, int filterDim){

	int k,l;                          //counting variables
	float sum=0.0;                          //temp sum
	int offset = ((filterDim-1)/2);     //bounds for inner loop

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	int i = y*imageWidth+ x ;

	for(k=0; k<filterDim; k++){
		for(l=0; l<filterDim ; l++){
			if((i+l-offset+(k-offset)*imageWidth > 0) && (i+l-offset+(k-offset)*imageWidth< imageWidth*imageHeight) && (i%imageWidth + l - offset > 0) && (i%imageWidth + l - offset < imageWidth) && (i%imageHeight + k - offset >0) && (i%imageHeight +k - offset < imageHeight))                        //COnditions if the filter falls over the image or off. First 2 check the width and last 2 check the heights
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
	printf("%d \n",imageWidth*imageHeight);
	for(i=0; i< imageWidth*imageHeight; i++){
		sum=0.0;
		for(k=0; k<filterDim; k++){
			for(l=0; l<filterDim ; l++){
				if((i+l-offset+(k-offset)*imageWidth > 0) && (i+l-offset+(k-offset)*imageWidth< imageWidth*imageHeight) && (i%imageWidth + l - offset > 0) && (i%imageWidth + l - offset < imageWidth) && (i%imageHeight + k - offset >0) && (i%imageHeight +k - offset < imageHeight))                        //COnditions if the filter falls over the image or off. First 2 check the width and last 2 check the heights
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
//	 float filter[] ={-1.0,0.0,1.0,-2.0,0.0,2.0,-1.0,0.0,1.0};            //highlights edges
	float filter[] ={-1.0,-1.0,-1.0,-1.0,9.0,-1.0,-1.0,-1.0,-1.0};      //sharpens image
//	float filter[] ={1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0};   //slightly blurs image

	int filterDim=FILTERDIM;  						//dimensions of the filter, assume its square
//======================================================

    	if (imagePath == NULL)
    	{
       	 printf("Unable to source image file: %s\n", imageFilename);
       	 exit(EXIT_FAILURE);
    	}

    	sdkLoadPGM(imagePath, &hData, &width, &height);             //load image into hdata and initialize width and height, hdata is a 1D array
			printf("%d %d \n", width,height);
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

//RUN THE CONSTANT MEMORY GPU implementation=======
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


	const size_t cblock_size = 256;                                 //initialise block size
	size_t cgrid_size = width*height / cblock_size;                   // calculate gride size

  	// deal with a possible partial final block
 	 if(width*height % block_size) ++cgrid_size;

	 cudaEventCreate(&launch_begin);
	 cudaEventCreate(&launch_end);
	 // record a CUDA event immediately before and after the kernel launch
	 cudaEventRecord(launch_begin,0);
	 // launch the kernel
	 GPUConstantConv<<<cgrid_size, cblock_size>>>(dcData,dconstantFilter,dcOutput,width,height,filterDim) ;          //Call the kernal
	 cudaEventRecord(launch_end,0);
	 cudaEventSynchronize(launch_end);
	 // measure the time (ms) spent in the kernel
	 //	 float time = 0;
	 		cudaEventElapsedTime(&time, launch_begin, launch_end);

	 // copy the result back to the host memory space
	 cudaMemcpy(hOutputData, dcOutput, size, cudaMemcpyDeviceToHost);
	 printf("GPU Const run time: %fms\n", time);

	 sdkSavePGM("Image_CONST_OUT.pgm",hOutputData,width,height);

//=================================================

//TEXTURE MEMORY IMPLEMENTATION ====================================
//float *dData = NULL;

	float *dtFilter = 0;
	cudaMalloc((void**)&dtFilter, filterDim*filterDim*sizeof(float));
	cudaMemcpy(dtFilter, filter, filterDim*filterDim*sizeof(float), cudaMemcpyHostToDevice);	   //copy the filter to the device

	 checkCudaErrors(cudaMalloc((void **) &dData, size));

	 // Allocate array and copy image data
	 cudaChannelFormatDesc channelDesc =
			 cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	 cudaArray *cuArray;
	 checkCudaErrors(cudaMallocArray(&cuArray,
																	 &channelDesc,
																	 width,
																	 height));
	 checkCudaErrors(cudaMemcpyToArray(cuArray,
																		 0,
																		 0,
																		 hData,
																		 size,
																		 cudaMemcpyHostToDevice));

	 // Set texture parameters
	 tex.addressMode[0] = cudaAddressModeWrap;
	 tex.addressMode[1] = cudaAddressModeWrap;
	 tex.filterMode = cudaFilterModeLinear;
	 tex.normalized = false;    // access with normalized texture coordinates

	 // Bind the array to the texture
	 checkCudaErrors(cudaBindTextureToArray(tex, cuArray, channelDesc));

	 dim3 dimBlock(8, 8, 1);
	 dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

	 checkCudaErrors(cudaDeviceSynchronize());
	 StopWatchInterface *timer = NULL;
	 sdkCreateTimer(&timer);
	 sdkStartTimer(&timer);

	 // Execute the kernel
	 GPUTextureConv<<<dimGrid, dimBlock, 0>>>(dData,dtFilter, width, height,filterDim);

	 // Check if kernel execution generated an error
	 getLastCudaError("Kernel execution failed");

	 checkCudaErrors(cudaDeviceSynchronize());
	 sdkStopTimer(&timer);
				//	(width *height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
	 sdkDeleteTimer(&timer);

	 // Allocate mem for the result on host side
	// float *hOutputData = (float *) malloc(size);
	 // copy result from device to host
	 checkCudaErrors(cudaMemcpy(hOutputData,
															dData,
															size,
															cudaMemcpyDeviceToHost));

/*float *dtOutput = 0;
float *dtFilter = 0;
float *dtData = 0;


cudaMalloc((void**)&dtOutput, size);
cudaMalloc((void**)&dtFilter, filterDim*filterDim*sizeof(float));       //assign the space required for the above arrays
cudaMalloc((void**)&dtData, size);

 if(dtOutput == 0 || dtFilter == 0|| dtData ==0)                       //check if the arrays actually initialised properly
	{
		printf("couldn't allocate device memory (texture)\n");
		return 1;
	}


	cudaChannelFormatDesc channelDesc =
			cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray *cuArray;
	checkCudaErrors(cudaMallocArray(&cuArray,
																	&channelDesc,
																	width,
																	height));
	checkCudaErrors(cudaMemcpyToArray(cuArray,
																		0,
																		0,
																		hData,
																		size,
																		cudaMemcpyHostToDevice));

	// Set texture parameters
	tex.addressMode[0] = cudaAddressModeWrap;
	tex.addressMode[1] = cudaAddressModeWrap;
	tex.filterMode = cudaFilterModeLinear;
	tex.normalized = true;    // access with normalized texture coordinates

	// Bind the array to the texture*
	checkCudaErrors(cudaBindTextureToArray(tex, cuArray, channelDesc));
//	checkCudaErrors(cudaMemcpy(dtData, hData, size, cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaBindTexture(NULL, tex, dtData,size));

		dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

		 cudaEventCreate(&launch_begin);
		 cudaEventCreate(&launch_end);
		 // record a CUDA event immediately before and after the kernel launch
		 cudaEventRecord(launch_begin,0);
     GPUTextureConv<<<dimGrid, dimBlock, 0>>>(dData,dtFilter, width, height,filterDim);
		 cudaEventRecord(launch_end,0);
		cudaEventSynchronize(launch_end);
		// measure the time (ms) spent in the kernel
			 cudaEventElapsedTime(&time, launch_begin, launch_end);

		// copy the result back to the host memory space
		cudaMemcpy(hOutputData, dtData, size, cudaMemcpyDeviceToHost);




	/*const size_t tblock_size = 256;                                 //initialise block size
	size_t tgrid_size = width*height / tblock_size;                   // calculate gride size

  	// deal with a possible partial final block
 	 if(width*height % block_size) ++tgrid_size;

	 cudaEventCreate(&launch_begin);
	 cudaEventCreate(&launch_end);
	 // record a CUDA event immediately before and after the kernel launch
	 cudaEventRecord(launch_begin,0);
	 // launch the kernel
	 GPUTextureConv<<<tgrid_size, tblock_size>>>(dtOutput,dtFilter,width,height,filterDim) ;          //Call the kernal
	 cudaEventRecord(launch_end,0);
	 cudaEventSynchronize(launch_end);
	 // measure the time (ms) spent in the kernel
	 		cudaEventElapsedTime(&time, launch_begin, launch_end);

	 // copy the result back to the host memory space
	 cudaMemcpy(hOutputData, dtOutput, size, cudaMemcpyDeviceToHost);*/
	 printf("GPU Texture run time: %fms\n", time);

	 sdkSavePGM("Image_TEXT_OUT.pgm",hOutputData,width,height);

//==================================================================



//	for(int i=0;i<width*height;i++)
//	printf("%f " , hOutputData[i]);	                         //testing purposes


//	sdkSavePGM("Image_OUT.pgm",hOutputData,width,height);               //save the new image as Image_out.pgm


}
