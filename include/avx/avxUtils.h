#ifndef _AVX_UTILS_H_
#define _AVX_UTILS_H_

#include <algorithm>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>

template <class T>
class Traits_avx{
 public:
  static const int size=-1;
 };

/*template <>
class Traits<int>{
  public:
    typedef __m128i m128;
    static const int size=sizeof(int);
    static inline __m128i _mm_load(int *src){
      return _mm_loadu_si128((__m128i*)src);
    }
    static inline void _mm_store(m128 *src, m128 a){
      _mm_storeu_si128(src,a);
    }
    template <int mask>
      static inline __m128i _mm_shuffle(__m128i lo , __m128i hi){
        __m128 loF,hiF,RF;
        loF = _mm_castsi128_ps(lo);
        hiF = _mm_castsi128_ps(hi);
        RF = _mm_shuffle_ps(loF,hiF,mask);
        return _mm_castps_si128(RF);
      }
    static inline __m128i _mm_min(__m128i a , __m128i b ){
      return _mm_min_epi32(a,b);
    }
    static inline __m128i _mm_max(__m128i a , __m128i b ){
      return _mm_max_epi32(a,b);
    }
    static inline __m128i _mm_unpacklo(__m128i a, __m128i b){
      return _mm_unpacklo_epi32(a,b);
    }
    static inline __m128i _mm_unpackhi(__m128i a, __m128i b){
      return _mm_unpackhi_epi32(a,b);
    }

};*/

/*template <>
class Traits<float>{
  public:
    typedef __m128 m128;
    static const int size=sizeof(float);
    static inline __m128 _mm_load(float* src){
      return _mm_loadu_ps(src);
    }
    static inline void _mm_store(m128 *src, m128 a){
      _mm_storeu_ps((float*)src,a);
    }
    template <int mask>
      static inline __m128 _mm_shuffle(__m128 lo , __m128 hi){
        return _mm_shuffle_ps(lo,hi,mask);
      }
    static inline __m128 _mm_min(__m128 a , __m128 b ){
      return _mm_min_ps(a,b);
    }
    static inline __m128 _mm_max(__m128 a , __m128 b ){
      return _mm_max_ps(a,b);
    }
    static inline __m128 _mm_unpacklo(__m128 a, __m128 b){
      return _mm_unpacklo_ps(a,b);
    }
    static inline __m128 _mm_unpackhi(__m128 a, __m128 b){
      return _mm_unpackhi_ps(a,b);
    }
};*/
/*template <>
class Traits<int>{
  public:
    typedef __m256i m256; 
    static const int size=sizeof(int);
	static inline __m256i _mm_load(int* src){
		return _mm256_loadu_si256((__m256i*)src);
	}
	static inline void _mm_store(__m256i* src, __m256i dst){
		_mm256_storeu_si256(src,dst);
	}
	template <int mask>
	  static inline __m256i _mm_shuffle(__m256i lo, __m256i hi){
		__m256 loF, hiF,rF;
		loF = _mm256_castsi256_ps(lo);
		hiF = _mm256_castsi256_ps(hi);
		rF = _mm256_shuffle_ps(loF,hiF,mask);
		return _mm256_castps_si256(rF);		
	  }	
	template <int mask>
	  static inline __m256i _mm_permute2(__m256i a, __m256i b){
		return _mm256_permute2f128_si256(a,b,mask);
	  }
	template <int mask>
	  static inline __m256i _mm_permute(__m256i a){
		__m256 aF,rF;
		aF = _mm256_castsi256_ps(a);
		rF = _mm256_permute_ps(aF,mask);
		return _mm256_castps_si256(rF);
	  }
	static inline __m256i _mm_min(__m256i a, __m256i b){
		__m256 aF, bF,rF;
		aF = _mm256_castsi256_ps(a);
		bF = _mm256_castsi256_ps(b);
		rF = _mm256_min_ps(aF,bF);
		return _mm256_castps_si256(rF);		
	}
	static inline __m256i _mm_max(__m256i a, __m256i b){
		__m256 aF, bF,rF;
		aF = _mm256_castsi256_ps(a);
		bF = _mm256_castsi256_ps(b);
		rF = _mm256_max_ps(aF,bF);
		return _mm256_castps_si256(rF);		
	}
	static inline __m256i _mm_unpacklo(__m256i a, __m256i b){
		__m256 aF, bF,rF;
		aF = _mm256_castsi256_ps(a);
		bF = _mm256_castsi256_ps(b);
		rF = _mm256_unpacklo_ps(aF,bF);
		return _mm256_castps_si256(rF);		
	}
	static inline __m256i _mm_unpackhi(__m256i a, __m256i b){
		__m256 aF, bF,rF;
		aF = _mm256_castsi256_ps(a);
		bF = _mm256_castsi256_ps(b);
		rF = _mm256_unpackhi_ps(aF,bF);
		return _mm256_castps_si256(rF);		
	}
    
};*/
template <>
class Traits_avx<float>{
  public:
    typedef __m256 m256; 
    static const int size=sizeof(float);
	static inline __m256 _mm_load(float* src){
		return _mm256_loadu_ps(src);
	}
    	static inline void _mm_store(__m256* src, __m256 dst){
      		_mm256_storeu_ps((float*)src,dst);
    	}
	template <int mask>
	  static inline __m256 _mm_shuffle(__m256 lo, __m256 hi){
	  	return _mm256_shuffle_ps(lo,hi,mask);
	  }	
	template <int mask>
	  static inline __m256 _mm_permute2(__m256 a, __m256 b){
		return _mm256_permute2f128_ps(a,b,mask);
	  }
	template <int mask>
	  static inline __m256 _mm_permute(__m256 a){
		return _mm256_permute_ps(a,mask);
	  }
	static inline __m256 _mm_min(__m256 a, __m256 b){
		return _mm256_min_ps(a,b);
	}
	static inline __m256 _mm_max(__m256 a, __m256 b){
		return _mm256_max_ps(a,b);
	}
	static inline __m256 _mm_unpacklo(__m256 a, __m256 b){
		return _mm256_unpacklo_ps(a,b);
	}
	static inline __m256 _mm_unpackhi(__m256 a, __m256 b){
		return _mm256_unpackhi_ps(a,b);
	}
    
};

/*
template <>
class Traits<long>{
  public:
    static const int size=sizeof(long);
};
*/

template <>
class Traits_avx<double>{
  public:
    typedef __m256d m256; 
    static const int size=sizeof(double);
	static inline __m256d _mm_load(double* src){
		return _mm256_loadu_pd(src);
	}
    	static inline void _mm_store(__m256d* src, __m256d dst){
      		_mm256_storeu_pd((double*)src,dst);
    	}
	template <int mask>
	  static inline __m256d _mm_shuffle(__m256d lo, __m256d hi){
	  	return _mm256_shuffle_pd(lo,hi,mask);
	  }	
	template <int mask>
	  static inline __m256d _mm_permute2(__m256d a, __m256d b){
		return _mm256_permute2f128_pd(a,b,mask);
	  }
	template <int mask>
	  static inline __m256d _mm_permute(__m256d a){
		return _mm256_permute_pd(a,mask);
	  }
	static inline __m256d _mm_min(__m256d a, __m256d b){
		return _mm256_min_pd(a,b);
	}
	static inline __m256d _mm_max(__m256d a, __m256d b){
		return _mm256_max_pd(a,b);
	}
	static inline __m256d _mm_unpacklo(__m256d a, __m256d b){
		return _mm256_unpacklo_pd(a,b);
	}
	static inline __m256d _mm_unpackhi(__m256d a, __m256d b){
		return _mm256_unpackhi_pd(a,b);
	}
    
};



// Generic.
template <class T, int size=Traits_avx<T>::size>
class avx{
  public:
    static void merge(T* A_, T* A_last,T* B_, T* B_last, T* C_){
      std::merge(A_,A_last, B_,B_last, C_);
    }
};

// Partial specialized template for size 4.
template <class T>
class avx<T,4>{
  public:
    typedef typename Traits_avx<T>::m256 m256;
    static void merge(T* A_, T* A_last,T* B_, T* B_last, T* C_);
};

// Partial specialized template for size 8.
template <class T>
class avx<T,8>{
 public:
  typedef typename Traits_avx<T>::m256 m256;
  static void merge(T* A_, T* A_last,T* B_, T* B_last, T* C_);
};

#include "avxUtils.txx"

#endif

