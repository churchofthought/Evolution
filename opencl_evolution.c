// rm /tmp/opencl_evolution;sudo killall -9 opencl_evolution;gcc opencl_evolution.c -std=c99 -framework OpenCL -lncurses -lcdk -pthread -Wall -fast -o /tmp/opencl_evolution; /tmp/opencl_evolution
// rm /tmp/opencl_evolution;sudo killall -9 opencl_evolution;gcc opencl_evolution.c -std=c99 -L/usr/local/cuda-6.5/lib64 -I/usr/local/cuda-6.5/include -Iinclude -lcuda -lOpenCL -lncurses -lcdk -lrt -pthread -Wall -Ofast -o /tmp/opencl_evolution; /tmp/opencl_evolution
#define EVO_VERSION 7.11
#define _GNU_SOURCE 1

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <signal.h>
#include <curses.h>
#include <time.h>
#include <locale.h>
#include <limits.h>
#include <pthread.h>
#include <cdk/cdk.h>
#include "opencl_defs.h"


#define REPEAT_TEXT_0(z, n, text) text
#define REPEAT_TEXT(count,text) BOOST_PP_REPEAT(count, REPEAT_TEXT_0, text)

#define REPEAT_BINARY_TEXT_FROM_TO_0(z, n, dat) BOOST_PP_SEQ_ELEM(0,dat) BOOST_PP_STRINGIZE(n) BOOST_PP_SEQ_ELEM(1,dat)
#define REPEAT_BINARY_TEXT_FROM_TO(from,to,bef,after) BOOST_PP_REPEAT_FROM_TO(from, to, REPEAT_BINARY_TEXT_FROM_TO_0, (bef)(after))


#define DEVLOG "[DEVICE %u] "
#define DEV_NAME_MAX 128


















static double startTime;

#ifdef __APPLE__
#include <mach/mach_time.h>
#include <OpenCL/opencl.h>
static double orwl_timebase = 1.E-9;
static inline double curtime() {
 return mach_absolute_time() * orwl_timebase;
}
static void init_time(){
	mach_timebase_info_data_t tb = { 0 };
	mach_timebase_info(&tb);
	orwl_timebase *= tb.numer;
	orwl_timebase /= tb.denom;
	startTime = curtime();
}
#else
#include <CL/cl.h>
#endif

#ifdef __linux__ 
static inline double curtime(){
	struct timespec time;
	clock_gettime(CLOCK_REALTIME, &time);
	return (time.tv_sec + time.tv_nsec * 1.E-9) ;
}
static void init_time(){
	startTime = curtime();
}
#endif

#ifdef _WIN32
#include <windows.h>
#include <Shlobj.h>
#define inline __inline
static inline double curtime(){
	return timeGetTime() / 1000;
}

static void init_time(){
	startTime = curtime();
}
#define SLEEP(x) Sleep(1000*x)
#define ftruncate _chsize
#define snprintf _snprintf
#define SIZET_FMT "I"
#define COMMA_FMT ""
#else
#define SIZET_FMT "z"
#define COMMA_FMT "'"
#define SLEEP(x) sleep(x)
#endif
















#define BASE_DIR "/home/evolution/Desktop/Evolution/"



#define INC_PATH BASE_DIR "include"

#define SETTINGS_FILE BASE_DIR "opencl_parms.txt"
#define KERNEL_FILE_PATH "/tmp/opencl_kernel.cu"
#define KERNEL_SRC_PATH BASE_DIR "opencl_kernel.c"
#define TICKERSINFO_FILE BASE_DIR "tickersInfo.txt"
#define TICKERS_FILE BASE_DIR "tickers.txt"
#define TICKERSDATA_FILE BASE_DIR "tickersData.dat"

#define DEBUG_FILE    BASE_DIR "opencl_debug.txt"

// #define MULTI_LINE_STRING(...) #__VA_ARGS__








#ifdef CUDA
	#include <cuda.h>
	#define CUDA_MODULE_PATH "/tmp/opencl_kernel.cubin"
	#define cl_device_id CUdevice
	#define cl_context CUcontext
	#define cl_program CUmodule
	#define cl_kernel CUfunction
	#define cl_command_queue CUstream
	#define cl_int CUresult
	#define cl_mem CUdeviceptr
#endif


#define CWD_SIZE 260


typedef struct user_data{
	cl_mem fitnessMem;
	cl_kernel clKernel;
	size_t globalWorkDimensions;
	size_t localWorkDimensions;
	unsigned int rounds;
	unsigned int deviceNum;
	#ifdef CUDA
		CUcontext clContext;
	#else
		cl_command_queue clCommandQueue;
		fitness_result fitnessResult;
	#endif
} user_data;

#ifndef CUDA
	static char* clKernelSourceCode;
#endif

static FILE* restrict fSettings;

static const char* restrict tickers;
static unsigned int tickersCount;
static unsigned int tickerDatapointCount;
static unsigned int tickersDataSize;

static 
#ifdef CUDA
	int 
#else
	unsigned int
#endif
	numDevices;

static user_data userDatas[8];


static float lastWrittenFitness;


static fitness_result curRes;
static pthread_mutex_t readMutex;







#define ENQUEUEMUTATIONBEGIN static inline void enqueueMutation(user_data* const restrict userData)
ENQUEUEMUTATIONBEGIN
;

#ifndef CUDA
	static char* strForDeviceType(cl_device_type deviceType){
		switch (deviceType){
			case CL_DEVICE_TYPE_ALL: return "CL_DEVICE_TYPE_ALL";
			case CL_DEVICE_TYPE_GPU: return "CL_DEVICE_TYPE_GPU";
			case CL_DEVICE_TYPE_CPU: return "CL_DEVICE_TYPE_CPU";
			case CL_DEVICE_TYPE_ACCELERATOR: return "CL_DEVICE_TYPE_ACCELERATOR";
		}
		return "N/A";
	}
#endif

#ifdef CUDA
	static char* cudaErrorStr(cl_int err){
		switch(err)
		{
			case CUDA_SUCCESS: return "OK";
			case CUDA_ERROR_INVALID_VALUE: return "invalid value";
			case CUDA_ERROR_OUT_OF_MEMORY: return "out of memory";
			case CUDA_ERROR_NOT_INITIALIZED: return "not initialized";
			case CUDA_ERROR_DEINITIALIZED: return "deinitialized";
			case CUDA_ERROR_NO_DEVICE : return "no supported device";
			case CUDA_ERROR_INVALID_DEVICE : return "invalid device";
			case CUDA_ERROR_INVALID_IMAGE : return "invalid image";
			case CUDA_ERROR_INVALID_CONTEXT : return "invalid context";
			case CUDA_ERROR_CONTEXT_ALREADY_CURRENT : return "context already current";
			case CUDA_ERROR_MAP_FAILED : return "map failed";
			case CUDA_ERROR_UNMAP_FAILED : return "unmap failed";
			case CUDA_ERROR_ARRAY_IS_MAPPED : return "array already mapped";
			case CUDA_ERROR_ALREADY_MAPPED : return "already mapped";
			case CUDA_ERROR_NO_BINARY_FOR_GPU : return "no binary for GPU";
			case CUDA_ERROR_ALREADY_ACQUIRED : return "already acquired";
			case CUDA_ERROR_NOT_MAPPED : return "not mapped";
			case CUDA_ERROR_INVALID_SOURCE : return "invalid source";
			case CUDA_ERROR_FILE_NOT_FOUND : return "file not found";
			case CUDA_ERROR_INVALID_HANDLE : return "invalid handle";
			case CUDA_ERROR_NOT_FOUND : return "not found";
			case CUDA_ERROR_NOT_READY : return "not ready";
			case CUDA_ERROR_LAUNCH_FAILED : return "launch failed";
			case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES : return "launch ot of resources";
			case CUDA_ERROR_LAUNCH_TIMEOUT : return "launch timeout";
			case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING : return "launch incompatible texturing";
			case CUDA_ERROR_UNKNOWN : return "CUDA_ERROR_UNKNOWN";
			case CUDA_ERROR_PROFILER_DISABLED: return "CUDA_ERROR_PROFILER_DISABLED";
			case CUDA_ERROR_PROFILER_NOT_INITIALIZED: return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";
			case CUDA_ERROR_PROFILER_ALREADY_STARTED: return "CUDA_ERROR_PROFILER_ALREADY_STARTED";
			case CUDA_ERROR_PROFILER_ALREADY_STOPPED: return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";
			case CUDA_ERROR_NOT_MAPPED_AS_ARRAY: return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";
			case CUDA_ERROR_NOT_MAPPED_AS_POINTER: return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";
			case CUDA_ERROR_ECC_UNCORRECTABLE: return "CUDA_ERROR_ECC_UNCORRECTABLE";
			case CUDA_ERROR_UNSUPPORTED_LIMIT: return "CUDA_ERROR_UNSUPPORTED_LIMIT";
			case CUDA_ERROR_CONTEXT_ALREADY_IN_USE: return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";
			case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED: return "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";
			case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";
			case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED: return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";
			case CUDA_ERROR_OPERATING_SYSTEM: return "CUDA_ERROR_OPERATING_SYSTEM";
			case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED: return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";
			case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED: return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";
			case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE: return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";
			case CUDA_ERROR_CONTEXT_IS_DESTROYED: return "CUDA_ERROR_CONTEXT_IS_DESTROYED";
			case CUDA_ERROR_ASSERT: return "CUDA_ERROR_ASSERT";
			case CUDA_ERROR_TOO_MANY_PEERS: return "CUDA_ERROR_TOO_MANY_PEERS";
			case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED: return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";
			case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED: return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";
			case CUDA_ERROR_NOT_PERMITTED: return "CUDA_ERROR_NOT_PERMITTED";
			case CUDA_ERROR_NOT_SUPPORTED: return "CUDA_ERROR_NOT_SUPPORTED";
			case CUDA_ERROR_INVALID_PTX: return "CUDA_ERROR_INVALID_PTX";
			case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT: return "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT";
			case CUDA_ERROR_ILLEGAL_ADDRESS: return "CUDA_ERROR_ILLEGAL_ADDRESS";
			case CUDA_ERROR_HARDWARE_STACK_ERROR: return "CUDA_ERROR_HARDWARE_STACK_ERROR";
			case CUDA_ERROR_ILLEGAL_INSTRUCTION: return "CUDA_ERROR_ILLEGAL_INSTRUCTION";
			case CUDA_ERROR_MISALIGNED_ADDRESS: return "CUDA_ERROR_MISALIGNED_ADDRESS";
			case CUDA_ERROR_INVALID_ADDRESS_SPACE: return "CUDA_ERROR_INVALID_ADDRESS_SPACE";
			case CUDA_ERROR_INVALID_PC: return "CUDA_ERROR_INVALID_PC";

		}
		return "unknown error";
	}
#else
	static char* clErrorStr(cl_int err) {
		switch (err) {
			case CL_SUCCESS:                            return "Success!";
			case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
			case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
			case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
			case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
			case CL_OUT_OF_RESOURCES:                   return "Out of resources";
			case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
			case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
			case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
			case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
			case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
			case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
			case CL_MAP_FAILURE:                        return "Map failure";
			case CL_INVALID_VALUE:                      return "Invalid value";
			case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
			case CL_INVALID_PLATFORM:                   return "Invalid platform";
			case CL_INVALID_DEVICE:                     return "Invalid device";
			case CL_INVALID_CONTEXT:                    return "Invalid context";
			case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
			case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
			case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
			case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
			case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
			case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
			case CL_INVALID_SAMPLER:                    return "Invalid sampler";
			case CL_INVALID_BINARY:                     return "Invalid binary";
			case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
			case CL_INVALID_PROGRAM:                    return "Invalid program";
			case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
			case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
			case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
			case CL_INVALID_KERNEL:                     return "Invalid kernel";
			case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
			case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
			case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
			case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
			case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
			case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
			case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
			case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
			case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
			case CL_INVALID_EVENT:                      return "Invalid event";
			case CL_INVALID_OPERATION:                  return "Invalid operation";
			case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
			case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
			case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
			default: return "Unknown";
		}
	}
#endif

static CDKSCREEN* cdkScreen;
static inline void suspendCurses(){
	fflush(stdout);
	setvbuf(stdout, 0, _IONBF, 0);
	endwin();
}
static inline void resumeCurses(){
	fflush(stdout);
	setvbuf(stdout, 0, _IONBF, 0);
	refresh();
}
static void initCurses(){
	cdkScreen = initCDKScreen(initscr());
	start_color();
	use_default_colors();
	init_pair(1, COLOR_RED, -1);
	init_pair(2, COLOR_GREEN, -1);
	init_pair(3, COLOR_BLUE, -1);
	init_pair(4, COLOR_YELLOW, -1);
	init_pair(5, COLOR_MAGENTA, -1);
	init_pair(6, COLOR_CYAN, -1);
	init_pair(7, COLOR_WHITE, -1);
	suspendCurses();
}

static void windDown(){
	destroyCDKScreen(cdkScreen);
	endCDK();
	suspendCurses();
	exit(0);
}

static inline void fillBuffRandomly(void* buf, const size_t size){
	FILE* f = fopen("/dev/urandom", "r");
	if (!f || !fread(buf, 1, size, f)){
		puts("Couldn't read from /dev/random!");
		windDown();
	}

	fclose(f);
}

#ifndef CUDA
	static void clNotify(const char *errinfo, const void *private_info, 
				  size_t cb, void *user_data){
		suspendCurses();
		printf("\nReceived OpenCL Notification: %s\n", errinfo);
	}
#endif

static inline double totalRounds(){
	const user_data* restrict userData;
	double rounds = 0;

	unsigned int z = numDevices;
	while (z--){
		userData = &userDatas[z];
		rounds += userData->rounds * userData->globalWorkDimensions * KERNEL_ITERATIONS;
	}

	return rounds;
}


#ifndef CUDA
	static inline void onCLReadComplete(cl_event event, cl_int event_command_exec_status, const user_data* restrict const userData){

		unsigned int z;
		const user_data* restrict udata;
		const fitness_result* restrict res = &userData->fitnessResult;

		if (res->fitness > curRes.fitness){
			pthread_mutex_lock(&readMutex);
			if (res->fitness > curRes.fitness){
				curRes = *res;
				for (z = numDevices; z--;){
					if (userData->deviceNum == z) continue;
					udata = &userDatas[z];
					clEnqueueWriteBuffer(
						udata->clCommandQueue, udata->fitnessMem, CL_FALSE,
						0, sizeof(fitness_result), res,
						0, 0, 0
					);
					clFlush(udata->clCommandQueue);
				}
			}
			pthread_mutex_unlock(&readMutex);
		}
	}
#endif

#ifdef CUDA
	static inline void cuEnqueueRead(const user_data* restrict const userData){
		unsigned int z;
		fitness_result res;
		const user_data* restrict udata;

		if (cuCtxSetCurrent(userData->clContext) != CUDA_SUCCESS){
			puts("Couldn't bind CUDA context to thread!");
			windDown();
		}

		for (;;){
			cuMemcpyDtoH(&res, userData->fitnessMem, sizeof(fitness_result));
			if (res.fitness > curRes.fitness){
				pthread_mutex_lock(&readMutex);
				if (res.fitness > curRes.fitness){
					curRes = res;
					for (z = numDevices; z--;){
						if (userData->deviceNum == z) continue;
						udata = &userDatas[z];
						cuCtxSetCurrent(udata->clContext);
						cuMemcpyHtoDAsync(udata->fitnessMem, &res, sizeof(fitness_result), NULL);
					}
					cuCtxSetCurrent(userData->clContext);
				}
				pthread_mutex_unlock(&readMutex);
			}
		}
	}
#endif

#ifndef CUDA
	static inline void onCLEnqueueComplete(cl_event event, cl_int event_command_exec_status, user_data* restrict userData){
		cl_event clReadEvent;
		++userData->rounds;

		clEnqueueReadBuffer(
			userData->clCommandQueue, userData->fitnessMem, CL_FALSE, 
			0, sizeof(fitness_result), &userData->fitnessResult, 
			0, 0, &clReadEvent);

		clSetEventCallback(clReadEvent, CL_COMPLETE, onCLReadComplete, userData);
		clReleaseEvent(clReadEvent);

		clFlush(userData->clCommandQueue);

		enqueueMutation(userData);
	}
#endif

static void skipline(FILE* f){
	int c;
	do
	  c = fgetc(f);
	while (c != '\n');
}

// static void file_put_contents(const char *filepath, const char *data)
// {
// 	FILE *fp = fopen(filepath, "w");
// 	fputs(data, fp);
// 	fclose(fp);
// }

static char* file_get_contents(const char *filepath)
{
	unsigned int len;
	char* buf;
	FILE *f = fopen(filepath, "r");
	fseek (f, 0, SEEK_END);
	len = ftell(f);
	buf = malloc(len + 1);
	buf[len] = 0x00;
	fseek (f, 0, SEEK_SET);
	if (!fread(buf, 1, len, f)){
		printf("Error freading in %s !", __func__);
		windDown();
	}
	fclose(f);

	return buf;
}

#ifdef CUDA
static void exec(char* cmd) {
	printf("Executing `%s`\n", cmd);
	fflush(stdout);

	char buffer[128];
	FILE* pipe = popen(cmd, "r");

	if (!pipe){
		printf("Error executing cmd: %s\n", cmd);
		windDown();
	}
	while(!feof(pipe))
		if(fgets(buffer, 128, pipe) != NULL)
			fputs(buffer, stdout);
	putchar('\n');

	pclose(pipe);
	fflush(stdout);
}
#endif

static void generateKernelSourceCode(user_data* userData){
	char* kernelSrc = file_get_contents(KERNEL_SRC_PATH);
	FILE* f = fopen(KERNEL_FILE_PATH, "w+");
	#ifndef CUDA
		unsigned int len = 
	#endif
	fprintf(
		f, kernelSrc,
		userData->globalWorkDimensions,
		userData->localWorkDimensions,
		tickerDatapointCount
	);
	#ifdef CUDA
		fclose(f);
		unlink(CUDA_MODULE_PATH);
		exec(NVCC_CMD);
	#else
		clKernelSourceCode = malloc(len+1);
		clKernelSourceCode[len] = 0x00;
		fseek (f, 0, SEEK_SET);
		if (!fread(clKernelSourceCode, 1, len, f)){
			printf("Error freading in %s !", __func__);
			windDown();
		}
		fclose(f);
	#endif
	free(kernelSrc);
}

static void fillDefaultParameters(){
	unsigned int z;
	
	for (z = NUM_MULTIPLIERS; z--;)
		curRes.parms.multipliers[z] = 0;
	
	for (z = NUM_EXPONENTS; z--;)
		curRes.parms.exponents[z] = 0;

	// curRes.parms.minGain = 0;
}

static void readSettings(){
	fSettings = fopen (SETTINGS_FILE , "r+");
	if (!fSettings){
		puts("No previous settings were found, using default parameters.");
		fillDefaultParameters();
		fSettings = fopen(SETTINGS_FILE, "w");
		if (!fSettings){
			printf("Could not create settings file, %s!", SETTINGS_FILE);
			windDown();
		}
		return;
	}
	fseek(fSettings, 0, SEEK_END);
	if (!ftell(fSettings)){
		fillDefaultParameters();
		return;
	}
	fseek(fSettings, 0, SEEK_SET);

	skipline(fSettings);
	if (!fscanf(fSettings, REPEAT_TEXT(NUM_MULTIPLIERS, "%e ") "\n", 
		BOOST_PP_ENUM_BINARY_PARAMS(NUM_MULTIPLIERS, &curRes.parms.multipliers[BOOST_PP_INC_,-1]BOOST_PP_INTERCEPT)
	)){
		printf("fscanf failed to read multipliers in %s!", __func__);
		windDown();
	}
	#if NUM_EXPONENTS > 0
		skipline(fSettings);
		if (!fscanf(fSettings, REPEAT_TEXT(NUM_EXPONENTS, "%i ") "\n", 
			BOOST_PP_ENUM_BINARY_PARAMS(NUM_EXPONENTS, &curRes.parms.exponents[BOOST_PP_INC_,-1]BOOST_PP_INTERCEPT)
		)){
			printf("fscanf failed to read exponents in %s!", __func__);
			windDown();
		}
	#endif
	// skipline(fSettings);
	// if (!fscanf(fSettings, "%e \n", 
	// 	&curRes.parms.minGain
	// )){
	// 	printf("fscanf failed to read minGain in %s!", __func__);
	// 	windDown();
	// }
}

static inline void writeSettings(){
	rewind(fSettings);
	if (ftruncate(fileno(fSettings), fprintf(fSettings, 
		"Multipliers:\n"
		REPEAT_TEXT(NUM_MULTIPLIERS, "%e ") "\n"
		#if NUM_EXPONENTS > 0
			"Exponents:\n"
			REPEAT_TEXT(NUM_EXPONENTS, "%i ") "\n"
		#endif
		// "MinGain:\n"
		// "%e \n"
		"Mathematica Format: Riffle[{"
		REPEAT_TEXT(NUM_MULTIPLIERS, "%e, ")
		"}[[;;-2]],{"
		REPEAT_TEXT(NUM_EXPONENTS, "%i, ")
		"}[[;;-2]]]\n"
		"Fitness:\n%f\n"
		,
		BOOST_PP_ENUM_BINARY_PARAMS(NUM_MULTIPLIERS, curRes.parms.multipliers[BOOST_PP_INC_,-1]BOOST_PP_INTERCEPT),
		#if NUM_EXPONENTS > 0
			BOOST_PP_ENUM_BINARY_PARAMS(NUM_EXPONENTS, curRes.parms.exponents[BOOST_PP_INC_,-1]BOOST_PP_INTERCEPT),
		#endif
		// curRes.parms.minGain,
		BOOST_PP_ENUM_BINARY_PARAMS(NUM_MULTIPLIERS, curRes.parms.multipliers[BOOST_PP_INC_,-1]BOOST_PP_INTERCEPT),
		#if NUM_EXPONENTS > 0
			BOOST_PP_ENUM_BINARY_PARAMS(NUM_EXPONENTS, curRes.parms.exponents[BOOST_PP_INC_,-1]BOOST_PP_INTERCEPT),
		#endif
		lastWrittenFitness = curRes.fitness
	))){
		printf("ftruncate failed in %s", __func__);
		windDown();
	}
	fflush(fSettings);
}

static void readTickersInfo(){
	FILE* f = fopen (TICKERSINFO_FILE , "r");
	if (!f){
		printf("Could not read %s!", TICKERSINFO_FILE);
		windDown();
	}
	skipline(f);
	if (!fscanf(f, "%u\n", &tickersCount)){
		printf("fscanf failed to read tickersCount in %s!", __func__);
		windDown();
	}
	skipline(f);
	if (!fscanf(f, "%u\n", &tickerDatapointCount)){
		printf("fscanf failed to read tickerDatapointCount in %s!", __func__);
		windDown();
	}
	// skipline(f);
	// fscanf(f, "%u\n", &floatsPerDatapoint);
	fclose(f);

	tickersDataSize = 
		  tickersCount 
		* tickerDatapointCount 
		* sizeof(data_point);
}

static data_point* readTickersData(){
	FILE* f;
	data_point* tickersData;

	printf("Reading in %s for %u tickers: %s...\n\n", TICKERSDATA_FILE, tickersCount, tickers);
	tickersData = malloc(tickersDataSize);
	f = fopen(TICKERSDATA_FILE, "rb");
	if (!f){
		printf("Could not read %s!", TICKERSDATA_FILE);
		windDown();
	}
	fseek(f, 0, SEEK_END);
	if (tickersDataSize != ftell(f)){
		fclose(f);
		printf("Error! %s should be %u bytes\n",
			TICKERSDATA_FILE
			, tickersDataSize);
		windDown();
	}
	fseek(f, 0, SEEK_SET); 
	if (!fread(tickersData, 1, tickersDataSize, f)){
		printf("Error freading in %s !", __func__);
		windDown();
	}
	fclose(f);

	return tickersData;
}

ENQUEUEMUTATIONBEGIN {
	#ifdef CUDA
		CUresult err;
		CUstream stream;
		const unsigned int globalGroups = userData->globalWorkDimensions / userData->localWorkDimensions;

		if (cuCtxSetCurrent(userData->clContext) != CUDA_SUCCESS){
			puts("Couldn't bind CUDA context to thread!");
			windDown();
		}
		if (cuStreamCreate(&stream, CUDA_STREAM_FLAGS) != CUDA_SUCCESS){
			puts("Error creating CUDA stream");
			windDown();
		}
		for (;;++userData->rounds){
			if ((err = cuLaunchKernel(userData->clKernel, 
				globalGroups,1,1,
				userData->localWorkDimensions,1,1,
				0, stream, 0, 0)) != CUDA_SUCCESS){
				printf("Error launching CUDA kernel (%s)!\n", cudaErrorStr(err));
				windDown();
			}
			cuStreamSynchronize(stream);
		}
	#else
		cl_event clEvent;

		clEnqueueNDRangeKernel(
			userData->clCommandQueue, userData->clKernel, 
			sizeof(userData->globalWorkDimensions) / sizeof(size_t), 0, 
			&userData->globalWorkDimensions, &userData->localWorkDimensions, 
			0, 0, &clEvent);

		clSetEventCallback(clEvent, CL_COMPLETE, onCLEnqueueComplete, userData);
		clReleaseEvent(clEvent);

		clFlush(userData->clCommandQueue);
	#endif
}

static void enqueueCurrentSpecies(cl_kernel kernel, user_data* userData){
	unsigned int z;
	user_data* udata;
	fitness_result res;
	cl_int errcode_ret;

	#ifdef CUDA
		errcode_ret = cuLaunchKernel(kernel, 
			1,1,1,
			1,1,1,
			0, 0, 0, 0);
		if (errcode_ret != CUDA_SUCCESS){
			printf("Error launching CUDA kernel: %s\n", cudaErrorStr(errcode_ret));
			windDown();
		}
		cuCtxSynchronize();
	#else
		errcode_ret = clEnqueueTask(
			userData->clCommandQueue, kernel,
			0, 0, 0
		);

		if (errcode_ret != CL_SUCCESS){
			printf("Error enqueueing kernel: %s\n", clErrorStr(errcode_ret));
			windDown();
		}

		clFinish(userData->clCommandQueue);
	#endif

	
 	#ifdef CUDA
		errcode_ret = cuMemcpyDtoH(&res, userData->fitnessMem, sizeof(fitness_result));
		if (errcode_ret != CUDA_SUCCESS){
			printf("Error memcpyingDtoH fitness memory: %s\n", cudaErrorStr(errcode_ret));
			windDown();
		}
	#else
		clEnqueueReadBuffer(
			userData->clCommandQueue, userData->fitnessMem, CL_TRUE, 
			0, sizeof(fitness_result), &res, 
			0, 0, 0
		);
	#endif
	
	curRes = res;
	for (z = numDevices; z--;){
		udata = &userDatas[z];
		#ifdef CUDA
			errcode_ret = cuMemcpyHtoD(udata->fitnessMem, &res, sizeof(fitness_result));
			if (errcode_ret != CUDA_SUCCESS){
				printf("Error memcpyingHtoD fitness memory: %s\n", cudaErrorStr(errcode_ret));
				windDown();
			}
		#else
			clEnqueueWriteBuffer(
				udata->clCommandQueue, udata->fitnessMem, CL_TRUE,
				0, sizeof(fitness_result), &res,
				0, 0, 0
			);
		#endif
	}
}

static void bootstrapDevice(
	#ifndef CUDA
		const cl_platform_id clPlatformID, 
	#endif
	const cl_device_id clDeviceID,
	user_data* const userData,
	data_point* tickersData
){
	cl_program clProgram;
	cl_int errcode_ret;
	cl_mem tickersDataMem;
	cl_mem seedMem;
	#ifdef MUTEX
		cl_mem mutexMem;
		int mutex = 0;
	#endif
	size_t seedBufSize;
	void* seedBuf;

	#ifndef CUDA
		cl_context clContext;
		size_t infoValues[3];
		cl_uint infoValue;
		cl_uint computeUnits;
		size_t maxWorkgroupSize;
		size_t preferredWorkgroupSizeMultiple;
		size_t deSize;
		char* deviceExtensions;

		size_t blSize;
		char* buildLog;
		char cwd[CWD_SIZE];
		char buildOptions[sizeof(cwd) + sizeof(CL_BUILD_INCLUDES) + sizeof(CL_BUILD_OPTIONS)];
	#endif

	cl_kernel curFitnessKernel;

	#ifdef CUDA
		int maxThreadsPerBlock;
		int maxBlockSize[3];
		int maxGridSize[3];
		int cudaVal;
		int processorCount;
		int maxThreadsPerProcessor;
		size_t dataByteSize;
	#else
		const cl_context_properties clContextProperties[] = {
			CL_CONTEXT_PLATFORM, (cl_context_properties)clPlatformID
		, 0};
	#endif

	#ifdef DEBUG
		debug_point* debugPoints;
		debug_point* debugPoint;
		cl_mem dbgMem;
		unsigned int i;
		FILE* dbgFile;
	#endif

	#ifdef CUDA
		errcode_ret = cuCtxCreate(&userData->clContext, CUDA_CTX_FLAGS, clDeviceID);
		if (errcode_ret == CUDA_SUCCESS)
			printf(DEVLOG"Successfully created CUDA context!\n", userData->deviceNum);
		else{
			printf(DEVLOG"Error creating CUDA Context (%s)!\n", userData->deviceNum, cudaErrorStr(errcode_ret));
			windDown();
		}

		errcode_ret = cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_L1);
		if (errcode_ret != CUDA_SUCCESS){
			printf(DEVLOG"Error setting cuda cache config (%s)!\n", userData->deviceNum, cudaErrorStr(errcode_ret));
			windDown();
		}

		cuDeviceGetAttribute(&maxGridSize[0], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, clDeviceID);
		cuDeviceGetAttribute(&maxGridSize[1], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, clDeviceID);
		cuDeviceGetAttribute(&maxGridSize[2], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, clDeviceID);
		cuDeviceGetAttribute(&maxBlockSize[0], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, clDeviceID);
		cuDeviceGetAttribute(&maxBlockSize[1], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, clDeviceID);
		cuDeviceGetAttribute(&maxBlockSize[2], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, clDeviceID);
		cuDeviceGetAttribute(&maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, clDeviceID);

		printf(DEVLOG"maxGridSize = {%i,%i,%i}\n", userData->deviceNum, 
			maxGridSize[0],
			maxGridSize[1],
			maxGridSize[2]);
		printf(DEVLOG"maxBlockSize = {%i,%i,%i}\n", userData->deviceNum, 
			maxBlockSize[0],
			maxBlockSize[1],
			maxBlockSize[2]);
		printf(DEVLOG"maxThreadsPerBlock = %i\n", userData->deviceNum, 
			maxThreadsPerBlock);

		cuDeviceGetAttribute(&cudaVal, CU_DEVICE_ATTRIBUTE_WARP_SIZE, clDeviceID);
		printf(DEVLOG"CU_DEVICE_ATTRIBUTE_WARP_SIZE=%i\n", userData->deviceNum, cudaVal);

		cuDeviceGetAttribute(&cudaVal, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, clDeviceID);
		printf(DEVLOG"CU_DEVICE_ATTRIBUTE_CLOCK_RATE=%i\n", userData->deviceNum, cudaVal);

		cuDeviceGetAttribute(&cudaVal, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, clDeviceID);
		printf(DEVLOG"CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK=%i\n", userData->deviceNum, cudaVal);

		cuDeviceGetAttribute(&cudaVal, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, clDeviceID);
		printf(DEVLOG"CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE=%i\n", userData->deviceNum, cudaVal);

		cuDeviceGetAttribute(&processorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, clDeviceID);
		printf(DEVLOG"CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT=%i\n", userData->deviceNum, processorCount);

		cuDeviceGetAttribute(&maxThreadsPerProcessor, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, clDeviceID);
		printf(DEVLOG"CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR=%i\n", userData->deviceNum, maxThreadsPerProcessor);

		cuDeviceGetAttribute(&cudaVal, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, clDeviceID);
		printf(DEVLOG"CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR=%i\n", userData->deviceNum, cudaVal);

		cuDeviceGetAttribute(&cudaVal, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, clDeviceID);
		printf(DEVLOG"CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR=%i\n", userData->deviceNum, cudaVal);

		#ifdef FORCE_LOCAL_THREADS
			userData->localWorkDimensions = FORCE_LOCAL_THREADS;
		#else
			userData->localWorkDimensions = maxThreadsPerBlock;
		#endif
		userData->globalWorkDimensions = GLOBAL_WORK_MULTIPLE * processorCount * maxThreadsPerProcessor;

		putchar('\n');

		printf(DEVLOG"Setting local_items = %" SIZET_FMT "u = maxThreadsPerBlock(%i)\n", userData->deviceNum, userData->localWorkDimensions, maxThreadsPerBlock);
		printf(DEVLOG"Setting global_groups = %" SIZET_FMT "u, global_items = %" SIZET_FMT "u = GLOBAL_WORK_MULTIPLE(%u) * processorCount(%i) * maxThreadsPerProcessor(%i)\n", userData->deviceNum, userData->globalWorkDimensions / userData->localWorkDimensions, userData->globalWorkDimensions, GLOBAL_WORK_MULTIPLE, processorCount, maxThreadsPerProcessor);
		printf(DEVLOG"Using kernel_iterations=%u\n", userData->deviceNum, KERNEL_ITERATIONS);

		putchar('\n');

		#ifdef CUDA_COMPILE_BIN
			if (userData->deviceNum == (numDevices - 1) ) {
				generateKernelSourceCode(userData);
			}
		#endif

		errcode_ret = cuModuleLoad(&clProgram, CUDA_MODULE_PATH);
		if (errcode_ret == CUDA_SUCCESS)
			printf(DEVLOG"Successfully loaded CUDA kernel module from %s\n", userData->deviceNum, CUDA_MODULE_PATH);
		else{
			printf(DEVLOG"Failed to load CUDA kernel module from %s (%s)!", userData->deviceNum, CUDA_MODULE_PATH, cudaErrorStr(errcode_ret));
			windDown();
		}

		errcode_ret = cuModuleGetFunction(&userData->clKernel, clProgram, "fitness");


		if (errcode_ret == CUDA_SUCCESS){
			printf(DEVLOG"Successfully retrieved CUDA kernel fitness function ref\n", userData->deviceNum);
		}else{
			printf(DEVLOG"Failed to retrieve CUDA kernel fitness function ref (%s)!", userData->deviceNum, cudaErrorStr(errcode_ret));
			windDown();
		}

		errcode_ret = cuFuncGetAttribute(
				&maxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, userData->clKernel
		);
		if (errcode_ret == CUDA_SUCCESS)
			printf(DEVLOG"kernel module CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK=%i, local_items=%"SIZET_FMT"u\n", userData->deviceNum, maxThreadsPerBlock, userData->localWorkDimensions);
		else{
			printf(DEVLOG"Failed to cuFuncGetAttribute kernel module (%s)!", userData->deviceNum, cudaErrorStr(errcode_ret));
			windDown();
		}
	#else
		clGetDeviceInfo(
			clDeviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE, 
			sizeof(maxWorkgroupSize), &maxWorkgroupSize, 0
		);
		printf(DEVLOG"CL_DEVICE_MAX_WORK_GROUP_SIZE=%"SIZET_FMT"u\n", userData->deviceNum, maxWorkgroupSize);

		clGetDeviceInfo(
			clDeviceID, CL_DEVICE_MAX_WORK_ITEM_SIZES, 
			sizeof(infoValues), &infoValues, 0
		);
		printf(DEVLOG"CL_DEVICE_MAX_WORK_ITEM_SIZES=(%"SIZET_FMT"u,%"SIZET_FMT"u,%"SIZET_FMT"u)\n", userData->deviceNum, infoValues[0], infoValues[1], infoValues[2]);

		clGetDeviceInfo(
			clDeviceID, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, 
			sizeof(infoValue), &infoValue, 0
		);
		printf(DEVLOG"CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS=%u\n", userData->deviceNum, infoValue);

		clGetDeviceInfo(
			clDeviceID, CL_DEVICE_MAX_COMPUTE_UNITS, 
			sizeof(computeUnits), &computeUnits, 0
		);
		printf(DEVLOG"CL_DEVICE_MAX_COMPUTE_UNITS=%u\n",  userData->deviceNum, computeUnits);

		clGetDeviceInfo(
			clDeviceID, CL_DEVICE_MAX_CLOCK_FREQUENCY, 
			sizeof(infoValue), &infoValue, 0
		);
		printf(DEVLOG"CL_DEVICE_MAX_CLOCK_FREQUENCY=%u\n\n", userData->deviceNum, infoValue);

		clGetDeviceInfo(
			clDeviceID, CL_DEVICE_EXTENSIONS, 
			0, 0, &deSize
		);

		deviceExtensions = malloc(deSize);

		clGetDeviceInfo(
			clDeviceID, CL_DEVICE_EXTENSIONS, 
			deSize, deviceExtensions, 0
		);
		printf(DEVLOG"CL_DEVICE_EXTENSIONS=%s\n", userData->deviceNum, deviceExtensions);
		free(deviceExtensions);

		clContext = clCreateContext(
			clContextProperties, 1, &clDeviceID, 
			&clNotify, 0, &errcode_ret
		);
		if (errcode_ret == CL_SUCCESS)
			printf(DEVLOG"Successfully created OpenCL context!\n", userData->deviceNum);
		else{
			printf(DEVLOG"Error creating OpenCL Context (%s)\n", userData->deviceNum, clErrorStr(errcode_ret));
			windDown();
		}

		generateKernelSourceCode(userData);
		clProgram = 
		clCreateProgramWithSource(
			clContext, 1, (const char**) &clKernelSourceCode, 0, &errcode_ret
		);
		free(clKernelSourceCode);
		if (errcode_ret == CL_SUCCESS)
			printf(DEVLOG"Successfully created OpenCL program!\n", userData->deviceNum);
		else{
			printf("Error creating OpenCL program from kernel source(%s)\n", clErrorStr(errcode_ret));
			windDown();
		}

		printf(DEVLOG"Building OpenCL program... \n", userData->deviceNum);

		if (!getcwd(cwd, sizeof(cwd))){
			printf(DEVLOG"Couldnt call getcwd in %s", userData->deviceNum, __func__);
			windDown();
		}
		sprintf(buildOptions, CL_BUILD_INCLUDES CL_BUILD_OPTIONS, cwd);

		errcode_ret = clBuildProgram(clProgram, 1, &clDeviceID, buildOptions, 0, 0);
		clGetProgramBuildInfo(
				clProgram, clDeviceID, CL_PROGRAM_BUILD_LOG, 0, 0, &blSize
			);
		buildLog = malloc(blSize);
		clGetProgramBuildInfo(
			clProgram, clDeviceID, CL_PROGRAM_BUILD_LOG, blSize, buildLog, 0
		);
		printf("\n%s\n", buildLog);
		free(buildLog);

		if (errcode_ret == CL_SUCCESS)
			printf(DEVLOG"Successfully built the OpenCL program!\n", userData->deviceNum);
		else{
			printf("Error building OpenCL program (%s) \n", clErrorStr(errcode_ret));
			windDown();
		}
	#endif
	#ifdef CUDA
		errcode_ret = cuModuleGetGlobal(&tickersDataMem, &dataByteSize, clProgram, "tickersData");
		if (errcode_ret == CUDA_SUCCESS){
			printf(DEVLOG"Successfully retrieved tickersData global memory from CUDA module!\n", userData->deviceNum);
		}else{
			printf("Error retrieving CUDA tickersData mem buffer (%s)\n", cudaErrorStr(errcode_ret));
			windDown();
		}
		if (dataByteSize != tickersDataSize){
			printf(DEVLOG"CUDA tickersData size(%" SIZET_FMT "u) != tickersDataSize(%u)\n", userData->deviceNum, dataByteSize, tickersDataSize);
			windDown();
		}
		errcode_ret = cuMemcpyHtoD(tickersDataMem, tickersData, tickersDataSize);
		if (errcode_ret == CUDA_SUCCESS){
			printf(DEVLOG"Successfully copied tickers data to CUDA tickers mem!\n", userData->deviceNum);
		}else{
			printf("Couldn't copy tickers data to CUDA tickers mem! (%s)\n", cudaErrorStr(errcode_ret));
			windDown();
		}
	#else 
		tickersDataMem = clCreateBuffer(
			clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
			tickersDataSize, 
			tickersData,
			&errcode_ret
		);

		if (errcode_ret == CL_SUCCESS)
			printf(DEVLOG"Successfully created tickersData OpenCL memory buffer!\n", userData->deviceNum);
		else{
			printf("Error creating OpenCL tickersData mem buffer (%s)\n", clErrorStr(errcode_ret));
			windDown();
		}
	#endif


	#ifndef CUDA
		userData->clCommandQueue = clCreateCommandQueue(
			clContext, clDeviceID, CL_QUEUE_FLAGS, &errcode_ret
		);

		if (errcode_ret != CL_SUCCESS){
			printf("Error creating OpenCL command queue (%s)\n", clErrorStr(errcode_ret));
			windDown();
		}

		userData->clKernel = clCreateKernel(clProgram, "fitness", &errcode_ret);
		if (errcode_ret != CL_SUCCESS){
			printf("Error creating the OpenCL kernel (%s)\n", clErrorStr(errcode_ret));
			windDown();
		}

		

		clGetKernelWorkGroupInfo(
			userData->clKernel, 0, 
			CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(preferredWorkgroupSizeMultiple), 
			&preferredWorkgroupSizeMultiple, NULL
		);

		printf(DEVLOG"CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE=%" SIZET_FMT "u\n", userData->deviceNum, preferredWorkgroupSizeMultiple);


		clGetKernelWorkGroupInfo(
			userData->clKernel, 0, 
			CL_KERNEL_WORK_GROUP_SIZE, sizeof(userData->localWorkDimensions), 
			&userData->localWorkDimensions, NULL);

		printf(DEVLOG"CL_KERNEL_WORK_GROUP_SIZE=%" SIZET_FMT "u\n", userData->deviceNum, userData->localWorkDimensions);


		userData->globalWorkDimensions = GLOBAL_WORK_MULTIPLE * userData->localWorkDimensions * computeUnits;
		printf(DEVLOG"Setting global_items = %"SIZET_FMT"u = GLOBAL_WORK_MULTIPLE(%u) * CL_KERNEL_WORK_GROUP_SIZE(%"SIZET_FMT"u) * computeUnits(%u)\n", userData->deviceNum, userData->globalWorkDimensions, GLOBAL_WORK_MULTIPLE, userData->localWorkDimensions, computeUnits);

		printf(DEVLOG"Using global_groups=%"SIZET_FMT"u global_items=%"SIZET_FMT"u local_items=%"SIZET_FMT"u\n", userData->deviceNum, userData->globalWorkDimensions/userData->localWorkDimensions, userData->globalWorkDimensions, userData->localWorkDimensions);
		printf(DEVLOG"Using kernel_iterations=%u\n", userData->deviceNum, KERNEL_ITERATIONS);
	#endif

	#ifdef CUDA
		errcode_ret = cuModuleGetGlobal(&userData->fitnessMem, &dataByteSize, clProgram, "fitnessResult");
		if (errcode_ret == CUDA_SUCCESS){
			printf(DEVLOG"Successfully retrieved fitnessResult global memory from CUDA module!\n", userData->deviceNum);
		}else{
			printf("Error retrieving CUDA fitnessResult mem buffer (%s)\n", cudaErrorStr(errcode_ret));
			windDown();
		}
		if (dataByteSize != sizeof(fitness_result)){
			printf(DEVLOG"CUDA fitnessResult size(%" SIZET_FMT "u) != sizeof(fitness_result)(%lu)\n", userData->deviceNum, dataByteSize, sizeof(fitness_result));
			windDown();
		}
		errcode_ret = cuMemcpyHtoD(userData->fitnessMem, &curRes, sizeof(fitness_result));
		if (errcode_ret == CUDA_SUCCESS){
			printf(DEVLOG"Successfully copied current fitness to CUDA fitness mem!\n", userData->deviceNum);
		}else{
			printf("Couldn't copy current fitness to CUDA fitness mem! (%s)\n", cudaErrorStr(errcode_ret));
			windDown();
		}
	#else
		userData->fitnessMem = clCreateBuffer(
			clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			sizeof(fitness_result),
			(void*) &curRes,
			&errcode_ret
		);

		if (errcode_ret != CL_SUCCESS){
			printf("Error creating OpenCL fitness mem buffer (%s)\n", clErrorStr(errcode_ret));
			windDown();
		}
	#endif

	seedBufSize = userData->globalWorkDimensions * sizeof(mwc64x_state_t);
	seedBuf = malloc(seedBufSize);

	printf(DEVLOG"Initializing random seed buffer[%.3f MB] from /dev/random...\n", userData->deviceNum, seedBufSize/10e6);
	fillBuffRandomly((void*) seedBuf, seedBufSize);

	#ifdef CUDA
		errcode_ret = cuModuleGetGlobal(&seedMem, &dataByteSize, clProgram, "seedMemory");
		if (dataByteSize != seedBufSize){
			printf(DEVLOG"CUDA seedBuf size(%" SIZET_FMT "u) != seedBufSize(%" SIZET_FMT "u)\n", userData->deviceNum, dataByteSize, seedBufSize);
			windDown();
		}
		if (errcode_ret == CUDA_SUCCESS){
			printf(DEVLOG"Successfully retrieved seedMemory global memory from CUDA module!\n", userData->deviceNum);
		}else{
			printf("Error retrieving CUDA seedMemory mem buffer (%s)\n", cudaErrorStr(errcode_ret));
			windDown();
		}
		errcode_ret = cuMemcpyHtoD(seedMem, seedBuf, seedBufSize);
		if (errcode_ret == CUDA_SUCCESS){
			printf(DEVLOG"Successfully memcopied random seed buffer to device seedMem buffer!\n", userData->deviceNum);
		}else{
			printf("Error retrieving CUDA seedMemory mem buffer (%s)\n", cudaErrorStr(errcode_ret));
			windDown();
		}

		#ifdef MUTEX
			errcode_ret = cuModuleGetGlobal(&mutexMem, &dataByteSize, clProgram, "mutex");
			if (dataByteSize != sizeof(int)){
				printf(DEVLOG"CUDA mutex size(%" SIZET_FMT "u) != sizeof(int)(%" SIZET_FMT "u)\n", userData->deviceNum, dataByteSize, sizeof(int));
				windDown();
			}
			if (errcode_ret == CUDA_SUCCESS){
				printf(DEVLOG"Successfully retrieved mutex global memory from CUDA module!\n", userData->deviceNum);
			}else{
				printf("Error retrieving CUDA seedMemory mem buffer (%s)\n", cudaErrorStr(errcode_ret));
				windDown();
			}
			errcode_ret = cuMemcpyHtoD(mutexMem, &mutex, sizeof(int));
			if (errcode_ret == CUDA_SUCCESS){
				printf(DEVLOG"Successfully memcopied mutex buffer to device mutex buffer!\n", userData->deviceNum);
			}else{
				printf("Error retrieving CUDA mutex mem buffer (%s)\n", cudaErrorStr(errcode_ret));
				windDown();
			}
		#endif
	#else
		seedMem = clCreateBuffer(
			clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			seedBufSize,
			seedBuf,
			&errcode_ret
		);

		if (errcode_ret != CL_SUCCESS){
			printf("Error creating OpenCL seed mem buffer (%s)\n", clErrorStr(errcode_ret));
			windDown();
		}

		#ifdef MUTEX
			mutexMem = clCreateBuffer(
				clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				sizeof(int),
				&mutex,
				&errcode_ret
			);

			if (errcode_ret != CL_SUCCESS){
				printf("Error creating OpenCL mutex mem buffer (%s)\n", clErrorStr(errcode_ret));
				windDown();
			}
		#endif

		if (clSetKernelArg(userData->clKernel, 0, sizeof(tickersDataMem), &tickersDataMem)
		!= CL_SUCCESS){
			printf("Error setting OpenCL kernel args (%s)\n", clErrorStr(errcode_ret));
			windDown();
		}
		if (clSetKernelArg(userData->clKernel, 1, sizeof(seedMem), &seedMem)
			!= CL_SUCCESS){
			printf("Error setting OpenCL kernel args (%s)\n", clErrorStr(errcode_ret));
			windDown();
		}
		if (clSetKernelArg(userData->clKernel, 2, sizeof(userData->fitnessMem), &userData->fitnessMem)
			!= CL_SUCCESS){
			printf("Error setting OpenCL kernel args (%s)\n", clErrorStr(errcode_ret));
			windDown();
		}
		#ifdef MUTEX
			if (clSetKernelArg(userData->clKernel, 3, sizeof(mutexMem), &mutexMem)
				!= CL_SUCCESS){
				printf("Error setting OpenCL kernel args (%s)\n", clErrorStr(errcode_ret));
				windDown();
			}
		#endif
	#endif


	if (userData->deviceNum == 0) {
		#ifdef CUDA
			errcode_ret = cuModuleGetFunction(&curFitnessKernel, clProgram, "curFitness");
			if (errcode_ret == CUDA_SUCCESS){
				printf(DEVLOG"Successfully retrieved CUDA kernel curFitness function ref\n", userData->deviceNum);
			}else{
				printf(DEVLOG"Failed to retrieve CUDA kernel curFitness function ref (%s)!", userData->deviceNum, cudaErrorStr(errcode_ret));
				windDown();
			}
		#else
			curFitnessKernel = clCreateKernel(clProgram, "curFitness", &errcode_ret);
			if (errcode_ret != CL_SUCCESS){
				printf("Error creating the OpenCL curFitness kernel (%s)\n", clErrorStr(errcode_ret));
				windDown();
			}
			if (clSetKernelArg(curFitnessKernel, 0, sizeof(tickersDataMem), &tickersDataMem)
			!= CL_SUCCESS){
				printf("Error setting OpenCL kernel args (%s)\n", clErrorStr(errcode_ret));
				windDown();
			}
			if (clSetKernelArg(curFitnessKernel, 1, sizeof(seedMem), &seedMem)
				!= CL_SUCCESS){
				printf("Error setting OpenCL kernel args (%s)\n", clErrorStr(errcode_ret));
				windDown();
			}
			if (clSetKernelArg(curFitnessKernel, 2, sizeof(userData->fitnessMem), &userData->fitnessMem)
				!= CL_SUCCESS){
				printf("Error setting OpenCL kernel args (%s)\n", clErrorStr(errcode_ret));
				windDown();
			}
			#ifdef MUTEX
				if (clSetKernelArg(curFitnessKernel, 3, sizeof(mutexMem), &mutexMem)
					!= CL_SUCCESS){
					printf("Error setting OpenCL kernel args (%s)\n", clErrorStr(errcode_ret));
					windDown();
				}
			#endif
		#endif

		#ifdef DEBUG
			debugPoints = calloc(tickerDatapointCount, sizeof(debug_point));

			#ifdef CUDA
				cuModuleGetGlobal(&dbgMem, &dataByteSize, clProgram, "debugPoints");
				cuMemcpyHtoD(dbgMem, debugPoints, tickerDatapointCount * sizeof(debug_point));
			#else
				dbgMem = clCreateBuffer(
					clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
					tickerDatapointCount * sizeof(debug_point),
					debugPoints,
					&errcode_ret
				);

				if (errcode_ret != CL_SUCCESS){
					printf("Error creating OpenCL debug memory buffer (%s)\n", clErrorStr(errcode_ret));
					windDown();
				}

				if (clSetKernelArg(curFitnessKernel, 3, sizeof(dbgMem), &dbgMem)
					!= CL_SUCCESS){
					printf("Error setting OpenCL kernel args (%s)\n", clErrorStr(errcode_ret));
					windDown();
				}
			#endif
		#endif

		enqueueCurrentSpecies(curFitnessKernel, userData);

		#ifdef DEBUG
			#ifdef CUDA
				cuMemcpyDtoH(debugPoints, dbgMem, tickerDatapointCount * sizeof(debug_point));
			#else
				if (clEnqueueReadBuffer(userData->clCommandQueue, dbgMem, CL_TRUE, 0, 
					tickerDatapointCount * sizeof(debug_point),
					debugPoints, 0, 0, 0) != CL_SUCCESS){
					puts("Couldn't read OpenCL debug memory buffer!");
					windDown();
				}
				clReleaseMemObject(dbgMem);
			#endif

			dbgFile = fopen(DEBUG_FILE, "w");

			for (i = 0; i < tickerDatapointCount; ++i){
				debugPoint = &debugPoints[i];
				fprintf(dbgFile, "%c point %u) day %u) price %f) %e\n", 
				debugPoint->transaction == BUY_TRANSACTION ? 'B' :
				debugPoint->transaction == SELL_TRANSACTION ? 'S' :
				' ',
				i, (unsigned int) tickersData[i].day, tickersData[i].price, debugPoint->polyVal);
			}

			fclose(dbgFile);

			printf("\nDEBUG: debug data written to %s\n\n", DEBUG_FILE);
		#endif

		#ifndef CUDA
			clReleaseKernel(curFitnessKernel);
		#endif

		#ifdef DEBUG
			free(debugPoints);
		#endif
	}

	free(seedBuf);
}

static void bootstrapCL(data_point* tickersData){
	#ifdef CUDA
		pthread_t thread;
	#else
		cl_uint numPlatforms;
		cl_platform_id* clPlatformIDs;
		cl_platform_id clPlatformID;
		CDKRADIO* platformRadio;
		size_t infoSize;
	#endif

	

	cl_device_id* clDeviceIDs;
	cl_device_id clDeviceID;
	unsigned int i;
	unsigned int z;

	#if !defined(ALL_DEVICES) || !defined(CUDA)
	unsigned int t;
	char** infoList;
	char* infoStr;
	CDKSELECTION* deviceSelection;
	char* DIALOG_SELECTION_LIST[2];
	#endif

	
	user_data* udata;

	#ifdef CUDA
		if (cuInit(0) == CUDA_SUCCESS)
			puts("CUDA has been initialized.");
		else{
			puts("Failed to initialize CUDA!");
			windDown();
		}
	#else
		clGetPlatformIDs(0,0,&numPlatforms);
		printf("Detected %u OpenCL platform(s).\n", numPlatforms);
	
		clPlatformIDs = malloc(sizeof(cl_platform_id) * numPlatforms);
		clGetPlatformIDs(numPlatforms, clPlatformIDs, 0);

		infoList = malloc(sizeof(char*) * numPlatforms);

		
		for (i = numPlatforms; i--;){
			clPlatformID = clPlatformIDs[i];

			clGetPlatformInfo(clPlatformID, CL_PLATFORM_NAME, 0, 0, &infoSize);

			infoStr = malloc(infoSize);
			clGetPlatformInfo(clPlatformID, CL_PLATFORM_NAME, infoSize, infoStr, 0);

			infoList[i] = infoStr;
		}

		resumeCurses();
		platformRadio = newCDKRadio(
			cdkScreen, CENTER, CENTER, 0, -8, -8, 
			"<C>Select OpenCL Platform", infoList, numPlatforms,
			'x' | A_BOLD | COLOR_PAIR(1), 0, A_BOLD | COLOR_PAIR(2), 1, 1
		);
		
		i = activateCDKRadio(platformRadio, 0);
		destroyCDKRadio(platformRadio);
		suspendCurses();
		

		if (i >= numPlatforms) windDown();

		clPlatformID = clPlatformIDs[i];

		

		printf("Using OpenCL Platform: %s\n\n", infoList[i]);
		
		for (i = numPlatforms; i--;)
			free(infoList[i]);
		free(infoList);
	#endif

	#ifdef CUDA
		cuDeviceGetCount(&numDevices);
		printf("Detected %u CUDA devices \n", numDevices);
	#else
		clGetDeviceIDs(clPlatformID, CL_DEVICE_TYPE_ALL, 0, 0, &numDevices);
		printf("Detected %u OpenCL devices ( ", numDevices);

		clGetDeviceIDs(clPlatformID, CL_DEVICE_TYPE_CPU, 0, 0, &numDevices);
		printf("%u CPUs, ", numDevices);

		clGetDeviceIDs(clPlatformID, CL_DEVICE_TYPE_GPU, 0, 0, &numDevices);
		printf("%u GPUs, ", numDevices);

		clGetDeviceIDs(clPlatformID, CL_DEVICE_TYPE_ACCELERATOR, 0, 0, &numDevices);
		printf("%u Accelerators)\n", numDevices);

		clGetDeviceIDs(clPlatformID, MAIN_DEVICE_TYPE, 0, 0, &numDevices);

		printf("Using %s (%u devices)\n", strForDeviceType(MAIN_DEVICE_TYPE), numDevices);
	#endif

	puts("\nPress enter to continue:");
	getchar();

	
	clDeviceIDs = malloc(sizeof(cl_device_id) * numDevices);

	#ifdef CUDA
		for (i = numDevices; i--;){
			cuDeviceGet(&clDeviceID, i);
			clDeviceIDs[i] = clDeviceID;
		}
	#else
		clGetDeviceIDs(clPlatformID, MAIN_DEVICE_TYPE, numDevices, clDeviceIDs, 0);
	#endif



	#ifndef ALL_DEVICES
		infoList = malloc(sizeof(char*) * numDevices);

		for (i = numDevices; i--;){
			clDeviceID = clDeviceIDs[i];
			#ifdef CUDA
				infoStr = malloc(DEV_NAME_MAX);
				cuDeviceGetName(infoStr, DEV_NAME_MAX, clDeviceID);
			#else
				clGetDeviceInfo(clDeviceID, CL_DEVICE_NAME, 0, 0, &infoSize);
				infoStr = malloc(infoSize);
				clGetDeviceInfo(clDeviceID, CL_DEVICE_NAME, infoSize, infoStr, 0);
			#endif

			infoList[i] = infoStr;
		}

		resumeCurses();
		DIALOG_SELECTION_LIST[0] = "    ";
		DIALOG_SELECTION_LIST[1] = "</01>[x] ";	
		deviceSelection = newCDKSelection(
			cdkScreen, CENTER, CENTER, 0, -8, -8, 
			"<C>Select "
			#ifdef CUDA
				"CUDA"
			#else
				"OpenCL"
			#endif
			" Devices", infoList, numDevices,
			DIALOG_SELECTION_LIST, 2, A_BOLD | COLOR_PAIR(2), 1, 1
		);

		activateCDKSelection(deviceSelection, 0);
		suspendCurses();

		z = 0;
		t = numDevices;

		for (i = 0; i < t; ++i)
			if (deviceSelection->selections[i]){
				printf("Using Device: %s\n", infoList[i]);
				clDeviceIDs[z++] = clDeviceIDs[i];
			}else
				--numDevices;

		if (numDevices <= 0) windDown();

		
		for(i = deviceSelection->listSize; i--;)
			free(infoList[i]);
		free(infoList);

		destroyCDKSelection(deviceSelection);
		suspendCurses();
	#endif

	putchar('\n');

	for (i = numDevices; i--;){
		udata = &userDatas[i];
		udata->rounds = 0;
		udata->deviceNum = i;
		bootstrapDevice(
			#ifndef CUDA
				clPlatformID, 
			#endif
			clDeviceIDs[i], &userDatas[i], tickersData
		);
	}

	lastWrittenFitness = curRes.fitness;

	printf("Starting fitness is %f\n", curRes.fitness);
	#ifdef DEBUG
		windDown();
	#endif
	puts("\nPress enter to start:");
	getchar();

	pthread_mutex_init(&readMutex, NULL);
	for (i = numDevices; i--;){
		for (z = QUEUE_LOAD; z--;){
			udata = &userDatas[i];
			printf(DEVLOG"Enqueueing job %u/%u...\n", i, z+1,QUEUE_LOAD);
			#ifdef CUDA
				pthread_create(&thread, NULL, (void * (*)(void *)) enqueueMutation, udata);
			#else
				enqueueMutation(udata);
			#endif
		}
		#ifdef CUDA
			pthread_create(&thread, NULL, (void * (*)(void *)) cuEnqueueRead, udata);
		#endif
		putchar('\n');
	}

	free(clDeviceIDs);
	free(tickersData);

	#ifndef CUDA
		free(clPlatformIDs);
	#endif
}

static inline void printStatus(){
	const double runningTime = (curtime() - startTime);
	const double rounds = totalRounds();

	clear();
	attron(COLOR_PAIR(4)); 
	attron(A_BOLD);
	addstr(
		"\n"
		"                      "
			#ifdef CUDA
				"CUDA"
			#else
				"OpenCL"
			#endif
		"\n"
		"              -- Light Evolution --\n"
		"	              v" BOOST_PP_STRINGIZE(EVO_VERSION) "\n\n"
	);
	attron(COLOR_PAIR(3)); 
	addstr("             Stock");
	attron(COLOR_PAIR(4)); 
	printw(" %s\n\n", tickers);
	attron(COLOR_PAIR(2));            
	addstr("            Rounds ");
	attroff(A_BOLD);
	printw("%"COMMA_FMT"16f \n",
		rounds
	);
	attron(COLOR_PAIR(1));
	attron(A_BOLD);
	addstr(
		"    Rounds Per Sec "
	);
	attroff(A_BOLD);
	printw("%"COMMA_FMT"16f / sec \n",
		rounds  / runningTime
	);
	attron(COLOR_PAIR(4));
	attron(A_BOLD);
	addstr(
		"      Running Time "

	);
	attroff(A_BOLD);
	printw("%"COMMA_FMT"16f sec \n",
		runningTime

	);
	attron(COLOR_PAIR(5));
	attron(A_BOLD);
	addstr("           Fitness ");
	attroff(A_BOLD);
	attron(A_STANDOUT);
	printw("%"COMMA_FMT"16f \n\n",
		curRes.fitness
	);
	attroff(A_STANDOUT);
	attron(COLOR_PAIR(6));
	attron(A_BOLD);
	addstr("                   Multipliers\n               ==================\n"
	);
	attroff(A_BOLD);
	printw(
		REPEAT_TEXT(BOOST_PP_DIV(NUM_MULTIPLIERS,PARMS_PER_ROW),"                " REPEAT_TEXT(PARMS_PER_ROW,"%"COMMA_FMT"16e") "\n") "\n\n",
		BOOST_PP_ENUM_BINARY_PARAMS(NUM_MULTIPLIERS, curRes.parms.multipliers[BOOST_PP_INC_,-1]BOOST_PP_INTERCEPT)
	);
	
	#if NUM_EXPONENTS > 0
		attron(COLOR_PAIR(7));
		attron(A_BOLD);
		addstr(
			"                    Exponents\n               ==================\n"
		);
		attroff(A_BOLD);
		printw(
			REPEAT_TEXT(BOOST_PP_DIV(NUM_EXPONENTS,4),"                %16i %16i %16i %16i\n") "\n\n",
			BOOST_PP_ENUM_BINARY_PARAMS(NUM_EXPONENTS, curRes.parms.exponents[BOOST_PP_INC_,-1]BOOST_PP_INTERCEPT)
		);
	#endif

	// attron(COLOR_PAIR(7));
	// attron(A_BOLD);
	// addstr(
	// 	"                     MinGain\n               ==================\n"
	// );
	// attroff(A_BOLD);
	// printw(
	// 	"                %"COMMA_FMT"16e\n",
	// 	curRes.parms.minGain
	// );	

	refresh();
}

static void changeToHomeDir(){
	#ifdef _WIN32
		char path[MAX_PATH];
		SHGetFolderPath(NULL,CSIDL_MYDOCUMENTS,NULL,SHGFP_TYPE_CURRENT,path);
	#else
		char* path = getenv("HOME");
	#endif

	if (chdir(path)){
		printf("chdir failed in %s!", __func__);
		windDown();
	}
}

int main() {
	// fflush(stdout);
	// #ifdef _WIN32
	// 	setvbuf(stdout, 0, _IOFBF, 4 * 1024);
	// #else
	// 	setvbuf(stdout, 0, _IONBF, 0);
	// #endif

	signal(SIGINT, windDown);

	setlocale(LC_ALL, "");

	initCurses();

	changeToHomeDir();

	readSettings();

	printf("\n   Light Evolution v%s\n----------------------------\n", BOOST_PP_STRINGIZE(EVO_VERSION));

	tickers = file_get_contents(TICKERS_FILE);
	readTickersInfo();

	bootstrapCL(readTickersData());

	
	resumeCurses();
	init_time();
	for (;;){
		printStatus();
		if (curRes.fitness != lastWrittenFitness)
			writeSettings();
		SLEEP(1);
	}

	return 0;
}








