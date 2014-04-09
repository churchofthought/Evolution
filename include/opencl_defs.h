#include <boost/preprocessor.hpp>

#define CUDA
// #define DEBUG
// #define MUTEX
#define ALL_DEVICES

#define QUEUE_LOAD 2

#define NUM_MULTIPLIERS 8
#define NUM_EXPONENTS 0
#define NUM_PARMS BOOST_PP_INC(BOOST_PP_ADD(NUM_MULTIPLIERS, NUM_EXPONENTS))

#define PARMS_PER_ROW 4

#ifdef CUDA
	#define FORCE_LOCAL_THREADS 640
	#define GLOBAL_WORK_MULTIPLE 1
	#define NVCC_CMD "/usr/local/cuda/bin/nvcc -I ."INC_PATH" -cubin "KERNEL_FILE_PATH" -odir /tmp/ -arch sm_35 -O3 -Xptxas -v 2>&1"
	#define CUDA_COMPILE_BIN
	#define CUDA_CTX_FLAGS 0
	#define CUDA_STREAM_FLAGS 0
#else
	#define GLOBAL_WORK_MULTIPLE 64
	#define CL_BUILD_INCLUDES "-I%s"INC_PATH" "
	#define CL_BUILD_OPTIONS "-cl-strict-aliasing"
	#define CL_QUEUE_FLAGS CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
	#define MAIN_DEVICE_TYPE CL_DEVICE_TYPE_ALL
#endif

// kernel config
#define KERNEL_ITERATIONS 1
// #define EXPONENTIAL_MUTATION_DISTRIBUTION_POW 2
// #define MAX_MULTIPLIER_VAL (UINT_MAX - 1)
// #define MAX_EXPONENT_VAL 4096
#define DTYPE double

typedef struct species_parms{
	float multipliers[NUM_MULTIPLIERS];
	int exponents[NUM_EXPONENTS];
	float minGain;
} species_parms;

typedef struct fitness_result {
	double fitness;
	species_parms parms;
} fitness_result;

typedef struct data_point {
	float day;
	float price;
} data_point;

#ifdef DEBUG
	#define BUY_TRANSACTION 1
	#define SELL_TRANSACTION 2
	typedef struct debug_point {
		double transaction;
		double polyVal;
	} debug_point;
#endif

typedef struct{ unsigned int x; unsigned int c; } mwc64x_state_t;