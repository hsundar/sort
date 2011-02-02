#include <cstdlib>
#include <iostream>
#include <vector>
#include <omp.h>
#include <sstream>
#include <ompUtils.h>
#include <parUtils.h>

#define __VERIFY__

using namespace std;

bool verify(long size);

int main(int argc, char **argv){

  // Initialize MPI
  MPI_Init(&argc, &argv);

  //Set number of OpenMP threads to use.
  omp_set_num_threads(atoi(argv[1]));

  // Find out my identity in the default communicator 
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  // Find out number of processes
  int p;
  MPI_Comm_size(MPI_COMM_WORLD,&p);

  //################ MPI WARMUP ###################
  std::vector<int> in_init(1); in_init.resize(1);
  std::vector<int> out_init;
  par::sampleSort(in_init,out_init, MPI_COMM_WORLD);
  //###############################################


  // Geerate random data
  long N_total = atol(argv[2]); //4000000;
  long N = N_total/p;
  srand(/*omp_get_wtime()+*/myrank+1);
  //N=rand()%N;
  int* A=new int[N];
  for(int i=0;i<N;i++)
    A[i]=rand();
  std::vector<int> in(&A[0],&A[N]);
  std::vector<int> out;


#ifdef __VERIFY__
  //Save input to file
  stringstream fname_;
  fname_<<"tmp/input_"<<myrank<<'\0';
  string fname=fname_.str();
  FILE* f=fopen(&fname[0],"wb+");
  fwrite(A,N,sizeof(int),f);
  fclose(f);
#endif


  double wtime;
  wtime = omp_get_wtime ( );
  par::sampleSort<int>(in, out, MPI_COMM_WORLD);
  wtime = omp_get_wtime ( ) - wtime;
  std::cout<<"P"<<myrank<<"    =>    Time:"<<wtime<<"    OMP_Threads:"<<omp_get_max_threads()<<'\n';


#ifdef __VERIFY__
  //Save output to file
  stringstream fname1_;
  fname1_<<"tmp/output_"<<myrank<<'\0';
  fname=fname1_.str();
  f=fopen(&fname[0],"wb+");
  fwrite(&out[0],out.size(),sizeof(int),f);
  fclose(f);

  verify(N_total);
#endif

  // Shut down MPI 
  MPI_Finalize();
  return 0;

}


bool verify(long size){

  // Find out my identity in the default communicator 
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  if(myrank!=0) return true;

  // Find out number of processes
  int p;
  MPI_Comm_size(MPI_COMM_WORLD,&p);

  int* A=new int[size];
  int* B=new int[size];

  long indx=0;
  for(int i=0;i<p;i++){
    stringstream fname_;
    fname_<<"tmp/input_"<<i<<'\0';
    string fname=fname_.str();
    FILE* f=fopen(&fname[0],"rb");

    fseek (f , 0 , SEEK_END);
    int N = ftell (f)/sizeof(int);
    rewind (f);

    fread(&A[indx],N,sizeof(int),f);
    indx+=N;
    fclose(f);
  }
  long N_total=indx;

  double wtime;
  wtime = omp_get_wtime ( );
  //omp_par::merge_sort(&A[0],&A[indx]);
  std::sort(&A[0],&A[N_total]);
  cout<<"SeqTime:"<<omp_get_wtime()-wtime<<'\n';

  indx=0;
  for(int i=0;i<p;i++){
    stringstream fname_;
    fname_<<"tmp/output_"<<i<<'\0';
    string fname=fname_.str();
    FILE* f=fopen(&fname[0],"rb");

    fseek (f , 0 , SEEK_END);
    int N = ftell (f)/sizeof(int);
    rewind (f);

    fread(&B[indx],N,sizeof(int),f);

    for(long j=indx;j<indx+N;j++){
      if(A[j]!=B[j]){
	cout<<A[j]<<" "<<B[j]<<'\n';
	cout<<"Failed at:"<<j<<'\n';
	return false;
      }
    }
    indx+=N;
    fclose(f);
  }
  if(indx!=N_total)
    std::cout<<"Output size is wrong.\n";

  return true;

}

