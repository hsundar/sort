#ifndef __HYPER_QUICKSORT_COMM_AVOID_H__
#define __HYPER_QUICKSORT_COMM_AVOID_H__

#include <cstdio>

#ifdef _PROFILE_SORT
  #include "sort_profiler.h"
#endif

namespace par {

template<typename T>
int RankSwapSort(std::vector<T>& arr, MPI_Comm comm_){ // O( ((N/p)+log(p))*(log(N/p)+log(p)) ) 
#ifdef __PROFILE_WITH_BARRIER__
  MPI_Barrier(comm);
#endif
  PROF_SORT_BEGIN
#ifdef _PROFILE_SORT
    total_sort.start();
#endif

  // Copy communicator. 
  MPI_Comm comm=comm_;

  // Get comm size and rank.
  int npes, myrank;
  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &myrank);

  if (!myrank) printf("==== rank-swap sort ==== \n");

  if(npes==1){
    if (!myrank) printf("npes == 1\n");

#ifdef _PROFILE_SORT
    seq_sort.start();
#endif        
    omp_par::merge_sort(&arr[0],&arr[arr.size()]);
#ifdef _PROFILE_SORT
    seq_sort.stop();
    total_sort.stop();
#endif        
    PROF_SORT_END
  }
  // buffers ... keeping all allocations together 
  std::vector<T>  commBuff;
  std::vector<T>  mergeBuff;
  std::vector<int> glb_splt_cnts(npes);
  std::vector<int> glb_splt_disp(npes,0);


  int omp_p=omp_get_max_threads();
  srand(myrank);

  // Local and global sizes. O(log p)
  DendroIntL totSize, nelem = arr.size(); assert(nelem);
  par::Mpi_Allreduce<DendroIntL>(&nelem, &totSize, 1, MPI_SUM, comm);
  DendroIntL nelem_ = nelem;
  if (!myrank) printf(" starting sequential sort\n");

  // Local sort.  O(n/p log n/p)
#ifdef _PROFILE_SORT
  seq_sort.start();
#endif			
  // omp_par::merge_sort(&arr[0], &arr[arr.size()]);
  std::sort(&arr[0], &arr[arr.size()]);
#ifdef _PROFILE_SORT
  seq_sort.stop();
#endif			
  if (!myrank) printf("finished sequential sort \n");

  // Binary split and merge in each iteration.
  while(npes>1 && totSize>0){ // O(log p) iterations.

    //Determine splitters. O( log(N/p) + log(p) )
#ifdef _PROFILE_SORT
    hyper_compute_splitters.start();
#endif				
    T split_key;
    DendroIntL totSize_new;
    //while(true)
    { 
      // Take random splitters. O( 1 ) -- Let p * splt_count = glb_splt_count = const = 100~1000
      int splt_count = (1000*nelem)/totSize; 
      if (npes>1000) 
        splt_count = ( ((float)rand()/(float)RAND_MAX)*totSize < (1000*nelem) ? 1 : 0 );

      if ( splt_count > nelem ) 
        splt_count = nelem;

      std::vector<T> splitters(splt_count);
      for(size_t i=0;i<splt_count;i++) 
        splitters[i]=arr[rand()%nelem];

      // Gather all splitters. O( log(p) )
      int glb_splt_count;

      par::Mpi_Allgather<int>(&splt_count, &glb_splt_cnts[0], 1, comm);
      omp_par::scan(&glb_splt_cnts[0],&glb_splt_disp[0],npes);

      glb_splt_count = glb_splt_cnts[npes-1] + glb_splt_disp[npes-1];

      std::vector<T> glb_splitters(glb_splt_count);

      MPI_Allgatherv(&splitters[0], splt_count, par::Mpi_datatype<T>::value(), 
          &glb_splitters[0], &glb_splt_cnts[0], &glb_splt_disp[0], 
          par::Mpi_datatype<T>::value(), comm);

      // Determine split key. O( log(N/p) + log(p) )
      std::vector<DendroIntL> disp(glb_splt_count,0);

      if(nelem>0){
#pragma omp parallel for
        for(size_t i=0;i<glb_splt_count;i++){
          disp[i]=std::lower_bound(&arr[0], &arr[nelem], glb_splitters[i]) - &arr[0];
        }
      }
      std::vector<DendroIntL> glb_disp(glb_splt_count,0);
      MPI_Allreduce(&disp[0], &glb_disp[0], glb_splt_count, par::Mpi_datatype<DendroIntL>::value(), MPI_SUM, comm);

      DendroIntL* split_disp = &glb_disp[0];
      for(size_t i=0; i<glb_splt_count; i++)
        if ( abs(glb_disp[i] - totSize/2) < abs(*split_disp - totSize/2) ) 
          split_disp = &glb_disp[i];
      split_key = glb_splitters[split_disp - &glb_disp[0]];

      totSize_new=(myrank<=(npes-1)/2?*split_disp:totSize-*split_disp);
      //double err=(((double)*split_disp)/(totSize/2))-1.0;
      //if(fabs(err)<0.01 || npes<=16) break;
      //else if(!myrank) std::cout<<err<<'\n';
    }
#ifdef _PROFILE_SORT
    hyper_compute_splitters.stop();
#endif

    // Split problem into two. O( N/p )
    bool swap_ranks = false;
    int partner;
    int split_id=(npes-1)/2;
    {
#ifdef _PROFILE_SORT
      hyper_communicate.start();
#endif				
      int new_p0 = (myrank<=split_id?0:split_id+1);
      int cmp_p0 = (myrank> split_id?0:split_id+1);
      int new_np = (myrank<=split_id? split_id+1: npes-split_id-1);
      int cmp_np = (myrank> split_id? split_id+1: npes-split_id-1);

      partner = myrank+cmp_p0-new_p0;
      if(partner>=npes) partner=npes-1;
      assert(partner>=0);

      // bool extra_partner=( npes%2==1  && npes-1==myrank );

      // Exchange send sizes.
      char *low_buff, *high_buff;
      char *sbuff, *lbuff; 

      int     rsizes[2],     ssizes[2], rsize=0, ssize=0, lsize;

      size_t split_indx=(nelem>0?std::lower_bound(&arr[0], &arr[nelem], split_key)-&arr[0]:0);
      
      ssizes[0] = split_indx;
      ssizes[1] = nelem - split_indx;
      
      low_buff  = (char *)(&arr[0]);
      high_buff = (char *)(&arr[split_indx]);

      MPI_Status status;
      MPI_Sendrecv (&ssizes, 2, MPI_INT, partner, 0,   &rsizes, 2, MPI_INT, partner, 0, comm, &status);
      
      // if(extra_partner) MPI_Sendrecv(&ext_ssize,1,MPI_INT,split_id,0,   &ext_rsize,1,MPI_INT,split_id,   0,comm,&status);

      // { modify for rank-swap
      if ( myrank > split_id ) {
        // default keep_high,
        if ( (ssizes[1] + rsizes[0]) < (ssizes[0] + rsizes[1]) ) {
          swap_ranks = true;
          ssize = ssizes[1];      
          sbuff = high_buff;
          lsize = ssizes[0];
          lbuff = low_buff;
          rsize = rsizes[0];      
        } else {
          ssize = ssizes[0];
          sbuff = low_buff;
          lsize = ssizes[1];
          lbuff = high_buff;
          rsize = rsizes[1];      
        }
      } else {
        // default keep_low
        if ( (ssizes[0] + rsizes[1]) < (ssizes[1] + rsizes[0]) ) {
          swap_ranks = true;
          ssize = ssizes[0];
          sbuff = low_buff;
          lsize = ssizes[1];
          lbuff = high_buff;
          rsize = rsizes[1];      
        } else {
          ssize = ssizes[1];      
          sbuff = high_buff;
          lsize = ssizes[0];
          lbuff = low_buff;
          rsize = rsizes[0];      
        }
      } 
      // } modify for rank-swap
      
      // Exchange data.
      commBuff.reserve(rsize/sizeof(T));
      char*     rbuff = (char *)(&commBuff[0]);
      MPI_Sendrecv (sbuff, ssize, MPI_BYTE, partner, 0, rbuff, rsize, MPI_BYTE, partner, 0, comm, &status);
#ifdef _PROFILE_SORT
      hyper_communicate.stop();
      hyper_merge.start();
#endif

      int nbuff_size=lsize+rsize;
      mergeBuff.reserve(nbuff_size/sizeof(T));
      char* nbuff= (char *)(&mergeBuff[0]);  // new char[nbuff_size];
      omp_par::merge<T*>((T*)lbuff, (T*)&lbuff[lsize], (T*)rbuff, (T*)&rbuff[rsize], (T*)nbuff, omp_p, std::less<T>());

      // Copy new data.
      totSize=totSize_new;
      nelem = nbuff_size/sizeof(T);
      mergeBuff.swap(arr);
#ifdef _PROFILE_SORT
      hyper_merge.stop();
#endif
    }

    {// Split comm.  O( log(p) ) ??
#ifdef _PROFILE_SORT
    hyper_comm_split.start();
#endif				
      MPI_Comm scomm;
      if (swap_ranks) {
        MPI_Comm_split(comm, partner<=split_id, partner, &scomm );
        comm   = scomm;
        npes   = (partner<=split_id? split_id+1: npes  -split_id-1);
        myrank = (partner<=split_id? partner    : partner-split_id-1);
      } else {  
        MPI_Comm_split(comm, myrank<=split_id, myrank, &scomm );
        comm   = scomm;
        npes   = (myrank<=split_id? split_id+1: npes  -split_id-1);
        myrank = (myrank<=split_id? myrank    : myrank-split_id-1);
      }
#ifdef _PROFILE_SORT
    hyper_comm_split.stop();
#endif				
    }
  }

  //! Consider swapping ranks back ...

  // SortedElem.resize(nelem);
  // SortedElem.assign(arr, &arr[nelem]);
  // if(arr_!=NULL) delete[] arr_;

  // par::partitionW<T>(SortedElem, NULL , comm_);
  //      par::partitionW<T>(arr, NULL , comm_);

#ifdef _PROFILE_SORT
  total_sort.stop();
#endif
  PROF_SORT_END
}//end function

//--------------------------------------------------------------------------------
template<typename T>
int RankSwapSort(std::vector<T>& arr, std::vector<T> & SortedElem, MPI_Comm comm_){ // O( ((N/p)+log(p))*(log(N/p)+log(p)) ) 
#ifdef __PROFILE_WITH_BARRIER__
  MPI_Barrier(comm);
#endif
  PROF_SORT_BEGIN
#ifdef _PROFILE_SORT
    total_sort.start();
#endif

  // Copy communicator.
  MPI_Comm comm=comm_;

  // Get comm size and rank.
  int npes, myrank, myrank_;
  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &myrank); myrank_=myrank;
  if(npes==1){
    // @dhairya isn't this wrong for the !sort-in-place case ... 
#ifdef _PROFILE_SORT
    seq_sort.start();
#endif        
    omp_par::merge_sort(&arr[0],&arr[arr.size()]);
#ifdef _PROFILE_SORT
    seq_sort.stop();
#endif        
    SortedElem  = arr;
#ifdef _PROFILE_SORT
    total_sort.stop();
#endif        
    PROF_SORT_END
  }

  int omp_p=omp_get_max_threads();
  srand(myrank);

  // Local and global sizes. O(log p)
  DendroIntL totSize, nelem = arr.size(); assert(nelem);
  par::Mpi_Allreduce<DendroIntL>(&nelem, &totSize, 1, MPI_SUM, comm);
  DendroIntL nelem_ = nelem;

  // Local sort.
#ifdef _PROFILE_SORT
  seq_sort.start();
#endif			
  T* arr_=new T[nelem]; memcpy (&arr_[0], &arr[0], nelem*sizeof(T));      
  omp_par::merge_sort(&arr_[0], &arr_[arr.size()]);
#ifdef _PROFILE_SORT
  seq_sort.stop();
#endif
  // Binary split and merge in each iteration.
  while(npes>1 && totSize>0){ // O(log p) iterations.

    //Determine splitters. O( log(N/p) + log(p) )
#ifdef _PROFILE_SORT
    hyper_compute_splitters.start();
#endif				
    T split_key;
    DendroIntL totSize_new;
    //while(true)
    { 
      // Take random splitters. O( 1 ) -- Let p * splt_count = glb_splt_count = const = 100~1000
      int splt_count=(1000*nelem)/totSize; 
      if(npes>1000) splt_count=(((float)rand()/(float)RAND_MAX)*totSize<(1000*nelem)?1:0);
      if(splt_count>nelem) splt_count=nelem;
      std::vector<T> splitters(splt_count);
      for(size_t i=0;i<splt_count;i++) 
        splitters[i]=arr_[rand()%nelem];

      // Gather all splitters. O( log(p) )
      int glb_splt_count;
      std::vector<int> glb_splt_cnts(npes);
      std::vector<int> glb_splt_disp(npes,0);
      par::Mpi_Allgather<int>(&splt_count, &glb_splt_cnts[0], 1, comm);
      omp_par::scan(&glb_splt_cnts[0],&glb_splt_disp[0],npes);
      glb_splt_count=glb_splt_cnts[npes-1]+glb_splt_disp[npes-1];
      std::vector<T> glb_splitters(glb_splt_count);
      MPI_Allgatherv(&    splitters[0], splt_count, par::Mpi_datatype<T>::value(), 
          &glb_splitters[0], &glb_splt_cnts[0], &glb_splt_disp[0], 
          par::Mpi_datatype<T>::value(), comm);

      // Determine split key. O( log(N/p) + log(p) )
      std::vector<DendroIntL> disp(glb_splt_count,0);
      if(nelem>0){
#pragma omp parallel for
        for(size_t i=0;i<glb_splt_count;i++){
          disp[i]=std::lower_bound(&arr_[0], &arr_[nelem], glb_splitters[i])-&arr_[0];
        }
      }
      std::vector<DendroIntL> glb_disp(glb_splt_count,0);
      MPI_Allreduce(&disp[0], &glb_disp[0], glb_splt_count, par::Mpi_datatype<DendroIntL>::value(), MPI_SUM, comm);

      DendroIntL* split_disp=&glb_disp[0];
      for(size_t i=0;i<glb_splt_count;i++)
        if( labs(glb_disp[i]-totSize/2) < labs(*split_disp-totSize/2)) split_disp=&glb_disp[i];
      split_key=glb_splitters[split_disp-&glb_disp[0]];

      totSize_new=(myrank<=(npes-1)/2?*split_disp:totSize-*split_disp);
      //double err=(((double)*split_disp)/(totSize/2))-1.0;
      //if(fabs(err)<0.01 || npes<=16) break;
      //else if(!myrank) std::cout<<err<<'\n';
    }
#ifdef _PROFILE_SORT
    hyper_compute_splitters.stop();
#endif

    // Split problem into two. O( N/p )
    int split_id=(npes-1)/2;
    {
#ifdef _PROFILE_SORT
      hyper_communicate.start();
#endif				

      int new_p0=(myrank<=split_id?0:split_id+1);
      int cmp_p0=(myrank> split_id?0:split_id+1);
      int new_np=(myrank<=split_id? split_id+1: npes-split_id-1);
      int cmp_np=(myrank> split_id? split_id+1: npes-split_id-1);

      int partner = myrank+cmp_p0-new_p0;
      if(partner>=npes) partner=npes-1;
      assert(partner>=0);

      bool extra_partner=( npes%2==1  && npes-1==myrank );

      // Exchange send sizes.
      char *sbuff, *lbuff;
      int     rsize=0,     ssize=0, lsize=0;
      int ext_rsize=0, ext_ssize=0;
      size_t split_indx=(nelem>0?std::lower_bound(&arr_[0], &arr_[nelem], split_key)-&arr_[0]:0);
      ssize=       (myrank> split_id? split_indx: nelem-split_indx )*sizeof(T);
      sbuff=(char*)(myrank> split_id? &arr_[0]   :  &arr_[split_indx]);
      lsize=       (myrank<=split_id? split_indx: nelem-split_indx )*sizeof(T);
      lbuff=(char*)(myrank<=split_id? &arr_[0]   :  &arr_[split_indx]);

      MPI_Status status;
      MPI_Sendrecv                  (&    ssize,1,MPI_INT, partner,0,   &    rsize,1,MPI_INT, partner,   0,comm,&status);
      if(extra_partner) MPI_Sendrecv(&ext_ssize,1,MPI_INT,split_id,0,   &ext_rsize,1,MPI_INT,split_id,   0,comm,&status);

      // Exchange data.
      char*     rbuff=              new char[    rsize]       ;
      char* ext_rbuff=(ext_rsize>0? new char[ext_rsize]: NULL);
      MPI_Sendrecv                  (sbuff,ssize,MPI_BYTE, partner,0,       rbuff,    rsize,MPI_BYTE, partner,   0,comm,&status);
      if(extra_partner) MPI_Sendrecv( NULL,    0,MPI_BYTE,split_id,0,   ext_rbuff,ext_rsize,MPI_BYTE,split_id,   0,comm,&status);
#ifdef _PROFILE_SORT
      hyper_communicate.stop();
      hyper_merge.start();
#endif
      int nbuff_size=lsize+rsize+ext_rsize;
      char* nbuff= new char[nbuff_size];
      omp_par::merge<T*>((T*)lbuff, (T*)&lbuff[lsize], (T*)rbuff, (T*)&rbuff[rsize], (T*)nbuff, omp_p, std::less<T>());
      if(ext_rsize>0 && nbuff!=NULL){
        char* nbuff1= new char[nbuff_size];
        omp_par::merge<T*>((T*)nbuff, (T*)&nbuff[lsize+rsize], (T*)ext_rbuff, (T*)&ext_rbuff[ext_rsize], (T*)nbuff1, omp_p, std::less<T>());
        if(nbuff!=NULL) delete[] nbuff; nbuff=nbuff1;
      }

      // Copy new data.
      totSize=totSize_new;
      nelem = nbuff_size/sizeof(T);
      if(arr_!=NULL) delete[] arr_; 
      arr_=(T*) nbuff; nbuff=NULL;

      //Free memory.
      if(    rbuff!=NULL) delete[]     rbuff;
      if(ext_rbuff!=NULL) delete[] ext_rbuff;
#ifdef _PROFILE_SORT
      hyper_merge.stop();
#endif				
    }

#ifdef _PROFILE_SORT
    hyper_comm_split.start();
#endif				
    {// Split comm.  O( log(p) ) ??
      MPI_Comm scomm;
      MPI_Comm_split(comm, myrank<=split_id, myrank, &scomm );
      comm=scomm;
      npes  =(myrank<=split_id? split_id+1: npes  -split_id-1);
      myrank=(myrank<=split_id? myrank    : myrank-split_id-1);
    }
#ifdef _PROFILE_SORT
    hyper_comm_split.stop();
#endif				
  }

  SortedElem.resize(nelem);
  SortedElem.assign(arr_, &arr_[nelem]);
  if(arr_!=NULL) delete[] arr_;

#ifdef _PROFILE_SORT
  sort_partitionw.start();
#endif
  //      par::partitionW<T>(SortedElem, NULL , comm_);
#ifdef _PROFILE_SORT
  sort_partitionw.stop();
#endif

#ifdef _PROFILE_SORT
  total_sort.stop();
#endif
  PROF_SORT_END
}//end function

};

#endif
