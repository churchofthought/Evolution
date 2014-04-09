#define MWC64X_A 4294883355U

#define RAND_FLOAT(seed) as_float(RAND(seed))
#define RAND_UFLOAT(seed) as_float(RAND(seed) % 0x80000000U)
#define RAND_UINT(max, seed) (RAND(seed) % (max))

static inline uint RAND(mwc64x_state_t* const seedVal){
	const uint Xn = MWC64X_A * seedVal->x + seedVal->c; 
	seedVal->c = mad_hi(MWC64X_A, seedVal->x, (uint)(Xn<seedVal->c));
	seedVal->x = Xn;
	return seedVal->x ^ seedVal->c;
}                                                          
	                                                        