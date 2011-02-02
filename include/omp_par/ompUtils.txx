#include <cstdlib>
#include <omp.h>
#include <iterator>
#include <vector>
#include <seqUtils.h>

template <class T,class StrictWeakOrdering>
void omp_par::merge(T A_,T A_last,T B_,T B_last,T C_,int p,StrictWeakOrdering comp){
  typedef typename std::iterator_traits<T>::difference_type _DiffType;
  typedef typename std::iterator_traits<T>::value_type _ValType;

  _DiffType N1=A_last-A_;
  _DiffType N2=B_last-B_;

  int n=10;
  std::vector<_ValType> split; split.resize(p*n*2);
  std::vector<_DiffType> split_size; split_size.resize(p*n*2);

  #pragma omp parallel for
  for(int i=0;i<p;i++){
    for(int j=0;j<n;j++){
      int indx=i*n+j;
      _DiffType indx1=indx*(N1/(p*n));
      split   [indx]=A_[indx1];
      split_size[indx]=indx1+seq::BinSearch(B_,B_last,split[indx],comp);

      indx1=indx*(N2/(p*n));
      indx+=p*n;
      split   [indx]=B_[indx1];
      split_size[indx]=indx1+seq::BinSearch(A_,A_last,split[indx],comp);
    }
  }

  std::vector<_ValType> split1; split1.resize(p+1);
  std::vector<_DiffType> split_size1; split_size1.resize(p+1);
  std::vector<_DiffType> split_indx_A; split_indx_A.resize(p+1);
  std::vector<_DiffType> split_indx_B; split_indx_B.resize(p+1);
  split_indx_A[0]=0;
  split_indx_B[0]=0;
  split_indx_A[p]=N1;
  split_indx_B[p]=N2;

  #pragma omp parallel for
  for(int i=1;i<p;i++){
    _ValType  split1     =split     [0];
    _DiffType split_size1=split_size[0];
    _DiffType req_size=i*(N1+N2)/p;
    for(int j=0;j<p*n*2;j++){
      if(abs(split_size[j]-req_size)<abs(split_size1-req_size)){
        split1     =split   [j];
        split_size1=split_size[j];
      }
    }
    split_indx_A[i]=seq::BinSearch(A_,A_last,split1,comp);
    split_indx_B[i]=seq::BinSearch(B_,B_last,split1,comp);
  }

  #pragma omp parallel for
  for(int i=0;i<p;i++){
    T C=C_+split_indx_A[i]+split_indx_B[i];
    seq::Merge(A_+split_indx_A[i],A_+split_indx_A[i+1],B_+split_indx_B[i],B_+split_indx_B[i+1],C,comp);
  }
}

template <class T,class StrictWeakOrdering>
void omp_par::merge_sort(T A,T A_last,StrictWeakOrdering comp){
  typedef typename std::iterator_traits<T>::difference_type _DiffType;
  typedef typename std::iterator_traits<T>::value_type _ValType;

  int p=omp_get_max_threads();

  _DiffType N=A_last-A; 
  std::vector<_DiffType> split;
  split.resize(p+1);
  split[p]=N;

  #pragma omp parallel for
  for(int id=0;id<p;id++){
    split[id]=(N/p)*id;
  }
  #pragma omp parallel for
  for(int id=0;id<p;id++){
    std::sort(A+split[id],A+split[id+1]);
  }
  
  std::vector<_ValType> B; B.reserve(N);
  _ValType* A_=&A[0];//T A_=A;
  _ValType* B_=&B[0];//T B_=B.begin();
  for(int j=1;j<p;j=j*2){
    for(int i=0;i<p;i=i+2*j){
      if(i+j<p){
	omp_par::merge(A_+split[i],A_+split[i+j],A_+split[i+j],A_+split[(i+2*j<=p?i+2*j:p)],B_+split[i],p,comp);
      }else{
	#pragma omp parallel for
	for(int k=split[i];k<split[p];k++)
	  B_[k]=A_[k];
      }
    }
    _ValType* tmp_swap=A_;
    A_=B_;
    B_=tmp_swap;
  }
  if(A_!=&A[0]){//if(A_!=A){
    #pragma omp parallel for
    for(int i=0;i<N;i++)
      A[i]=A_[i];
  }
}

template <class T>
void omp_par::merge_sort(T A,T A_last){
  typedef typename std::iterator_traits<T>::value_type _ValType;
  omp_par::merge_sort(A,A_last,std::less<_ValType>());
}

template <class T, class I>
T omp_par::reduce(T* A, I cnt){
  T sum=0;
  #pragma omp parallel for reduction(+:sum)
  for(I i = 0; i < cnt; i++)
    sum+=A[i];
  return sum;
}

template <class T, class I>
void omp_par::scan(T* A, T* B,I cnt){
  int p=omp_get_max_threads();
  if(cnt<100*p){
    for(I i=1;i<cnt;i++)
      B[i]=B[i-1]+A[i-1];
    return;
  }

  I step_size=cnt/p;

  #pragma omp parallel for
  for(int i=0; i<p; i++){
    int start=i*step_size;
    int end=start+step_size;
    if(i==p-1) end=cnt;
    if(i!=0)B[start]=0;
    for(I j=start+1; j<end; j++)
      B[j]=B[j-1]+A[j-1];
  }

  T* sum=new T[p];
  sum[0]=0;
  for(int i=1;i<p;i++)
    sum[i]=sum[i-1]+B[i*step_size-1]+A[i*step_size-1];

  #pragma omp parallel for
  for(int i=1; i<p; i++){
    int start=i*step_size;
    int end=start+step_size;
    if(i==p-1) end=cnt;
    T sum_=sum[i];
    for(I j=start; j<end; j++)
      B[j]+=sum_;
  }

}


