#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// #pragma OPENCL EXTENSION cl_nv_pragma_unroll : enable

#include <opencl_defs.h>

#define BOOST_POW__N(z, n, text) BOOST_PP_EXPR_IF(n, *) text
#define BOOST_POW(x, y) (BOOST_PP_REPEAT(y, BOOST_POW__N, x))

// TODO use __restrict in the source code

#define GLOBAL_WORK_ITEMS %u
#define LOCAL_WORK_ITEMS %u
#define NUM_DATAPOINTS %u

#ifdef CUDA
	#define static __device__ static
	#define __constant __device__ __constant__
	#define __global __device__
	#define __kernel extern "C" __global__
	#define restrict __restrict__
	// #define inline __forceinline__
	#define as_float __int_as_float
	#define as_uint __float_as_int
	#define get_global_id(t) (blockIdx.x * LOCAL_WORK_ITEMS + threadIdx.x)
	#define mad_hi(a,b,c) (__umulhi(a,b) + (c))
	#define native_sqrt sqrtf
	#define native_powr powf
	#define atomic_cmpxchg atomicCAS
	#define mem_fence(t) __threadfence()
#else
	// #define inline 
#endif

#include <random.cl>

#ifdef CUDA
	__constant data_point tickersData[NUM_DATAPOINTS];
	__global mwc64x_state_t seedMemory[GLOBAL_WORK_ITEMS];
	__global fitness_result fitnessResult;
	#ifdef MUTEX
		__global int mutex;
	#endif
	#ifdef DEBUG
		__global debug_point debugPoints[NUM_DATAPOINTS];
	#endif
#endif

// #ifdef CUDA
// 	#define POW pow
// #else
#if NUM_EXPONENTS > 0
	static inline DTYPE POW(DTYPE base, int exp)
	{
	    DTYPE result = 1.;

	    if (exp < 0){
	    	exp = -exp;
	    	base = 1. / base;
	    }

	    while (exp) {
	        if (exp & 1)
	            result *= base;
	        exp >>= 1;
	        base *= base;
	    }

	    return result;
	}
#endif
// #endif

// static inline DTYPE POW(DTYPE base, int exp)
// {
//     DTYPE result = 1.;
//     if (exp < 0){
//     	exp = -exp;
//     	base = 1 / base;
//     }

//     while (exp--) result *= base;

//     return result;
// }

static inline void mutateParms(
	species_parms* restrict const parms,
	mwc64x_state_t* restrict const seedVal
){
	// const uint randParm = RAND_UINT(NUM_PARMS, seedVal);

	uint parmIndexes[NUM_PARMS] = {
		BOOST_PP_ENUM_PARAMS(NUM_PARMS,)
	};

	const uint parmsToChange = 
		#ifdef EXPONENTIAL_MUTATION_DISTRIBUTION_POW
			NUM_PARMS - (uint) native_powr((float)RAND_UINT(BOOST_POW(NUM_PARMS, EXPONENTIAL_MUTATION_DISTRIBUTION_POW), seedVal), 1. / MUTATION_DISTRIBUTION_POW);
		#else
			RAND_UINT(NUM_PARMS + 1, seedVal);
		#endif

	for (uint i = 0; i < parmsToChange; ++i){
		uint* const randParmPtr = &parmIndexes[NUM_PARMS - 1 - RAND_UINT(NUM_PARMS - i, seedVal)];
		const uint randParm = *randParmPtr;
		
		// if (randParm < NUM_MULTIPLIERS)
			parms->multipliers[randParm] = RAND_UFLOAT(seedVal);
		// else
		// 	parms->minGain = RAND(seedVal) * (1. / 4294967295.);
			
		// 	((int*)parms)[randParm] = (int) RAND_UINT(2 * MAX_MULTIPLIER_VAL + 1, seedVal) - MAX_MULTIPLIER_VAL;
		// else
		// 	((int*)parms)[randParm]  = (int) RAND_UINT(2 * MAX_EXPONENT_VAL + 1, seedVal) - MAX_EXPONENT_VAL;

		*randParmPtr = parmIndexes[i];
	}


}

static inline void evalFitness(
	#ifndef CUDA
		__constant const data_point* restrict const tickersData,
		__global fitness_result* restrict const fitnessResult,
		#ifdef MUTEX
			__global int* restrict const mutex,
		#endif
	#endif
	const species_parms* restrict const parms
	
	#ifndef CUDA
		#ifdef DEBUG
			,__global debug_point* const debugPoints
		#endif
	#endif
){
	#ifndef CUDA
		#define fitnessResult (*fitnessResult)
		#define mutex (*mutex)
	#endif

	
	unsigned int buyCount = 0;
	DTYPE fitnessVal;

	{
		DTYPE spent = 0., sold = 0., isHolding = 0.;

		for (uint i = NUM_MULTIPLIERS; i < NUM_DATAPOINTS; ++i){
			DTYPE mean = 0.;
			
			#ifndef CUDA
				__constant
			#endif
			const data_point* restrict const day = &tickersData[ i ];

			{
				DTYPE multiplierSum = 0.;

				// #ifndef CUDA
				// __constant
				// #endif
				// const data_point* restrict curDay = day;

				for (uint n = 0; n < NUM_MULTIPLIERS; ++n){
					#ifndef CUDA
						__constant
					#endif
					const data_point* restrict const otherDay = &tickersData[ i - n - 1];
					const DTYPE multiplier = parms->multipliers[ n ];
					
					multiplierSum += multiplier * (day->day - otherDay->day);

					mean += multiplier * day->price / otherDay->price;

					// curDay = otherDay;
				}

				mean /= multiplierSum;
			}


			#ifdef DEBUG
				debugPoints[i].polyVal = mean;
			#endif
			// if (mean >= parms->minGain) 
			if (mean >= MIN_GAIN) 
			{
				if (!isHolding){
					spent += (isHolding = day->price);
					++buyCount;
					#ifdef DEBUG
						debugPoints[i].transaction = BUY_TRANSACTION;
					#endif
				}
			}else if (isHolding){
				sold += day->price;
				isHolding = 0.;
				#ifdef DEBUG
					debugPoints[i].transaction = SELL_TRANSACTION;
				#endif
			}
		}

		fitnessVal = (spent > isHolding 
						? sold / (spent - isHolding)
						: 1. 
					) + buyCount * .01;
	}
			
	if (
		// isfinite(fitnessVal) && 
		fitnessVal > fitnessResult.fitness
	){
		#ifdef MUTEX
			while (atomic_cmpxchg(&mutex, 0, 1)){}
			if (fitnessVal > fitnessResult.fitness){
		#endif
		#ifdef CUDA
			fitnessResult = (fitness_result){
				fitnessVal, *parms
			};
		#else
			fitnessResult.fitness = fitnessVal;
			fitnessResult.parms = *parms;
		#endif
		#ifdef MUTEX
			}
			mutex = 0;
			mem_fence(CLK_GLOBAL_MEM_FENCE);
		#endif
	}

	#undef mutex
	#undef fitnessResult
}

 __kernel
 void fitness(
	#ifndef CUDA
		__constant const data_point* restrict const tickersData,
		__global mwc64x_state_t* restrict const seedMemory,
		__global fitness_result* restrict const fitnessResult
		#ifdef MUTEX
			,__global int* restrict const mutex
		#endif
		#ifdef DEBUG
			,__global debug_point* const debugPoints
		#endif
	#endif
) {

	#ifndef CUDA
		__global 
	#endif
	mwc64x_state_t* restrict const seed = &seedMemory[get_global_id(0)];
	mwc64x_state_t seedVal = *seed;


	#ifndef CUDA
		#define fitnessResult (*fitnessResult)
	#endif

	species_parms parms = fitnessResult.parms;
	mutateParms(&parms, &seedVal);

	#undef fitnessResult

	// #pragma unroll
	// for (uint i = KERNEL_ITERATIONS; i--;)
		evalFitness(
			#ifndef CUDA
				tickersData, 
				fitnessResult,
				#ifdef MUTEX
					mutex,
				#endif
				&parms
				#ifdef DEBUG
					,debugPoints
				#endif
			#else
				&parms
			#endif

		);

	*seed = seedVal;	
}

 __kernel
 void curFitness(
	#ifndef CUDA
		__constant data_point* restrict const tickersData,
		__global mwc64x_state_t* restrict const seedMemory,
		__global fitness_result* restrict const fitnessResult
		#ifdef MUTEX
			,__global int* const mutex
		#endif
		#ifdef DEBUG
			,__global debug_point* const debugPoints
		#endif
	#endif
) {
 	#ifndef CUDA
		#define fitnessResult (*fitnessResult)
	#endif

	species_parms parms = fitnessResult.parms;

	#undef fitnessResult

	evalFitness(
		#ifndef CUDA
			tickersData, fitnessResult, 
		#endif
		#ifdef MUTEX
			mutex, 
		#endif
		&parms
		#ifndef CUDA
			#ifdef DEBUG
				,debugPoints
			#endif
		#endif
	);
}