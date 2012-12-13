#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <omp.h>
#include <sstream>
#include <ompUtils.h>
#include <parUtils.h>
#include <octUtils.h>

#include <TreeNode.h>
#include <gensort.h>
#include <sortRecord.h>

#define MAX_DEPTH 30
//#define __VERIFY__

long getNumElements(char* code) {
  unsigned int slen = strlen(code);
  char dtype = code[0];
  char tmp[128];
  strncpy(tmp, code+1, slen-3); tmp[slen-3] = '\0';
  // std::cout << "tmp is " << tmp << std::endl;
  long numBytes = atoi(tmp);
  switch(code[slen-2]) {
    case 'g':
    case 'G':
      numBytes *= 1024*1024*1024;
      break;
    case 'k':
    case 'K':
      numBytes *= 1024;
      break;
    case 'm':
    case 'M':
      numBytes *= 1024*1024;
      break;
    default:
      // std::cout << "unknown code " << code[slen-2] << std::endl;
      return 0;
  };

  switch (dtype) {
    case 'i': // int array
      // std::cout << "Int : " << sizeof(int) << std::endl;
      return numBytes/sizeof(int);
      break;
    case 'l': // long array
      // std::cout << "Long : " << sizeof(long) << std::endl;
      return numBytes/sizeof(long);
      break;
    case 't': // treenode
      // std::cout << "TreeNode : " << sizeof(ot::TreeNode) << std::endl;
      return numBytes/sizeof(ot::TreeNode);
      break;
    case 'x': // gensort record
      // std::cout << "sortRecord : " << sizeof(sortRecord) << std::endl;
      return numBytes/sizeof(sortRecord);
      break;
    default:
      return 0;
  };

}

template <class T>
bool verify(std::vector<T>& in_, std::vector<T> &out_, MPI_Comm comm){

  // Find out my identity in the default communicator 
  int myrank, p;
  MPI_Comm_rank(comm, &myrank);
  MPI_Comm_size(comm,&p);

  std::vector<T> in;
  {
    int N_local=in_.size()*sizeof(T);
    std::vector<int> r_size(p, 0);
    std::vector<int> r_disp(p, 0);
    MPI_Gather(&N_local  , 1, MPI_INT, 
        &r_size[0], 1, MPI_INT, 0, comm);
    omp_par::scan(&r_size[0], &r_disp[0], p);

    if(!myrank) in.resize((r_size[p-1]+r_disp[p-1])/sizeof(T));
    MPI_Gatherv((char*)&in_[0],    N_local,             MPI_BYTE, 
        (char*)&in [0], &r_size[0], &r_disp[0], MPI_BYTE, 0, comm);
  }

  std::vector<T> out;
  {
    int N_local=out_.size()*sizeof(T);
    std::vector<int> r_size(p, 0);
    std::vector<int> r_disp(p, 0);
    MPI_Gather(&N_local  , 1, MPI_INT, 
        &r_size[0], 1, MPI_INT, 0, comm);
    omp_par::scan(&r_size[0], &r_disp[0], p);

    if(!myrank) out.resize((r_size[p-1]+r_disp[p-1])/sizeof(T));
    MPI_Gatherv((char*)&out_[0],    N_local,             MPI_BYTE, 
        (char*)&out [0], &r_size[0], &r_disp[0], MPI_BYTE, 0, comm);
  }

  if(in.size()!=out.size()){
    std::cout<<"Wrong size: in="<<in.size()<<" out="<<out.size()<<'\n';
    return false;
  }
  std::sort(&in[0], &in[in.size()]);

  for(long j=0;j<in.size();j++)
    if(in[j]!=out[j]){
      std::cout<<"Failed at:"<<j<<"; in="<<in[j]<<" out="<<out[j]<<'\n';
      return false;
    }

  return true;
}

std::vector<double> time_sort_bench(size_t N, MPI_Comm comm) {
  int myrank, p;

  MPI_Comm_rank(comm, &myrank);
  MPI_Comm_size(comm,&p);

  srand(myrank+1);

  typedef sortRecord Data_t;
  std::vector<Data_t> in(N);
  genRecords((char* )&(*(in.begin())), myrank, N);
  
  std::vector<Data_t> in_cpy=in;
  std::vector<Data_t> out;

  // Warmup run and verification.
  par::HyperQuickSort<Data_t>(in, out, comm);
  in=in_cpy;
#ifdef __VERIFY__
  verify(in,out,comm);
#endif
  
  std::vector<double> tt(3);
  double wtime;

  //Sort
  MPI_Barrier(comm);
  wtime=-omp_get_wtime();
  par::HyperQuickSort<Data_t>(in, out, comm);
  MPI_Barrier(comm);
  wtime+=omp_get_wtime();
  tt[0]=wtime;
  in=in_cpy;

  return tt;
}

std::vector<double> time_sort_tn(size_t N, MPI_Comm comm) {
  int myrank, p;

  MPI_Comm_rank(comm, &myrank);
  MPI_Comm_size(comm,&p);

  srand(myrank+1);

  typedef ot::TreeNode Data_t;
  std::vector<Data_t> in(N);
  unsigned int s = (1u << MAX_DEPTH);
  for(int i=0;i<N;i++){
    ot::TreeNode node(rand()%s, rand()%s, rand()%s, MAX_DEPTH-1, 3, MAX_DEPTH);
    in[i]=node; 
  }
  
  std::vector<Data_t> in_cpy=in;
  std::vector<Data_t> out;

  // Warmup run and verification.
  par::HyperQuickSort<Data_t>(in, out, comm);
  in=in_cpy;
#ifdef __VERIFY__
  verify(in,out,comm);
#endif
  
  std::vector<double> tt(3);
  double wtime;

  //Sort
  MPI_Barrier(comm);
  wtime=-omp_get_wtime();
  par::HyperQuickSort<Data_t>(in, out, comm);
  MPI_Barrier(comm);
  wtime+=omp_get_wtime();
  tt[0]=wtime;
  in=in_cpy;

  return tt;
}

template <class T>
std::vector<double> time_sort(size_t N, MPI_Comm comm){

  // Find out my identity in the default communicator 
  int myrank;
  MPI_Comm_rank(comm, &myrank);

  // Find out number of processes
  int p;
  MPI_Comm_size(comm,&p);

  // Geerate random data
  srand(/*omp_get_wtime()+*/myrank+1);

  // typedef int Data_t;
  std::vector<T> in(N);
  for(int i=0;i<N;i++){
    in[i]=rand(); 
  } // */
  std::vector<T> in_cpy=in;
  std::vector<T> out;

  // Warmup run and verification.
  par::HyperQuickSort<T>(in, out, comm);
  in=in_cpy;
#ifdef __VERIFY__
  verify(in,out,comm);
#endif

  /*
     par::bitonicSort<Data_t>(in, comm); out=in;
     in=in_cpy;
  #ifdef __VERIFY__
    verify(in,out,comm);
  #endif

  par::sampleSort<Data_t>(in, out, comm);
  in=in_cpy;
  #ifdef __VERIFY__
    verify(in,out,comm);
  #endif
  */
  std::vector<double> tt(3);
  double wtime;

  //Sort
  MPI_Barrier(comm);
  wtime=-omp_get_wtime();
  par::HyperQuickSort<T>(in, out, comm);
  MPI_Barrier(comm);
  wtime+=omp_get_wtime();
  tt[0]=wtime;
  in=in_cpy;
  /*
     MPI_Barrier(comm);
     wtime=-omp_get_wtime();
     par::bitonicSort<Data_t>(in, comm); out=in;
     MPI_Barrier(comm);
     wtime+=omp_get_wtime();
     tt[1]=wtime;
     in=in_cpy;

     MPI_Barrier(comm);
     wtime=-omp_get_wtime();
     par::sampleSort<Data_t>(in, out, comm);
     MPI_Barrier(comm);
     wtime+=omp_get_wtime();
     tt[2]=wtime;
     in=in_cpy;
     */
  return tt;
}

int main(int argc, char **argv){
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " numThreads typeSize" << std::endl;
    std::cerr << "\t\t typeSize is a character for type of data follwed by data size per node." << std::endl;
    std::cerr << "\t\t Examples:" << std::endl;
    std::cerr << "\t\t i1GB : integer  array of size 1GB" << std::endl;
    std::cerr << "\t\t l1GB : long     array of size 1GB" << std::endl;
    std::cerr << "\t\t t1GB : TreeNode array of size 1GB" << std::endl;
    std::cerr << "\t\t x4GB : 100byte  array of size 4GB" << std::endl;
    return 1;  
  }

  std::cout<<setiosflags(std::ios::fixed)<<std::setprecision(4)<<std::setiosflags(std::ios::right);

  // Initialize MPI
  MPI_Init(&argc, &argv);

  //Set number of OpenMP threads to use.
  omp_set_num_threads(atoi(argv[1]));

  // Find out my identity in the default communicator 
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  // Find out number of processes
  int p;
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  int proc_group=0;
  int min_np=1;
  MPI_Comm comm;
  for(int i=p;myrank<i && i>=min_np ;i=i>>1) proc_group++;
  MPI_Comm_split(MPI_COMM_WORLD, proc_group, myrank, &comm);

  std::vector<double> tt(3000000,0);
  
  int k = 0; // in case size based runs are needed 
  char dtype = argv[2][0];
  long N = getNumElements(argv[2]);
  if (!N) {
    std::cerr << "illegal typeSize code provided: " << argv[2] << std::endl;
    return 2;
  }

  // std::cout << "local array of size " << N << " of type " << dtype << std::endl;

  // check if arguments are ok ...
    
  { // -- full size run  
    std::vector<double> ttt;
    
    switch(dtype) {
      case 'i':
        ttt = time_sort<int>(N, MPI_COMM_WORLD);
        break;
      case 'l':
        ttt = time_sort<long>(N, MPI_COMM_WORLD);
        break;
      case 't':
        ttt = time_sort_tn(N, MPI_COMM_WORLD);
        break;
      case 'x':
        ttt = time_sort_bench(N, MPI_COMM_WORLD);
        break;
    };
    if(!myrank){
      tt[0*1000000+100*k+0]=ttt[0];
      tt[1*1000000+100*k+0]=ttt[1];
      tt[2*1000000+100*k+0]=ttt[2];
    }
  }
  { // smaller /2^k runs 
    int myrank_;
    MPI_Comm_rank(comm, &myrank_);
    std::vector<double> ttt;


    // ttt=time_sort(N, comm);
    switch(dtype) {
      case 'i':
        ttt = time_sort<int>(N, comm);
        break;
      case 'l':
        ttt = time_sort<long>(N, comm);
        break;
      case 't':
        ttt = time_sort_tn(N, comm);
        break;
      case 'x':
        ttt = time_sort_bench(N, comm);
        break;
    };

    if(!myrank_){
      tt[0*1000000+100*k+proc_group]=ttt[0];
      tt[1*1000000+100*k+proc_group]=ttt[1];
      tt[2*1000000+100*k+proc_group]=ttt[2];
    }
  }

  std::vector<double> tt_glb(3000000);
  MPI_Reduce(&tt[0], &tt_glb[0], 3000000, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if(!myrank){
    std::cout<<"\nNew Sort:\n";
    for(int i=0;i<proc_group;i++){
      int np=p;
      if(i>0) np=(p>>(i-1))-(p>>i);
      std::cout<<"\tP="<<np<<' ';
      // for(int k=0;k<=log_N;k++)
        std::cout<<tt_glb[0*1000000+100*k+i]<<' ';
      std::cout<<'\n';
    }
    /*
    std::cout<<"\n\nBitonic Sort:\n";
    for(int i=0;i<proc_group;i++){
      int np=p;
      if(i>0) np=(p>>(i-1))-(p>>i);
      std::cout<<"P="<<np<<' ';
      // for(int k=0;k<=log_N;k++)
        std::cout<<tt_glb[1*1000000+100*k+i]<<' ';
      std::cout<<'\n';
    }
    std::cout<<"\n\nOld Sort:\n";
    for(int i=0;i<proc_group;i++){
      int np=p;
      if(i>0) np=(p>>(i-1))-(p>>i);
      std::cout<<"P="<<np<<' ';
      // for(int k=0;k<=log_N;k++)
        std::cout<<tt_glb[2*1000000+100*k+i]<<' ';
      std::cout<<'\n';
    }
    std::cout<<"\n\n";
    */
  }

  // Shut down MPI 
  MPI_Finalize();
  return 0;

}

