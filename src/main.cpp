#include <iomanip>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <omp.h>
#include <sstream>
#include <ompUtils.h>
#include <parUtils.h>
#include <octUtils.h>
#include <TreeNode.h>
//#define __VERIFY__

using namespace std;

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


std::vector<double> time_sort(size_t N, MPI_Comm comm){
  // Find out my identity in the default communicator 
  int myrank;
  MPI_Comm_rank(comm, &myrank);

  // Find out number of processes
  int p;
  MPI_Comm_size(comm,&p);

  //################ MPI WARMUP ###################
  std::vector<int> in_init(1); in_init.resize(1);
  std::vector<int> out_init;
  //par::sampleSort(in_init,out_init, comm);
  //###############################################

  // Geerate random data
  srand(/*omp_get_wtime()+*/myrank+1);
  std::vector<ot::TreeNode> in(N);
  #define MAX_DEPTH 30
  unsigned int s=(1u << MAX_DEPTH);
  for(int i=0;i<N;i++){
    ot::TreeNode node(rand()%s, rand()%s, rand()%s, MAX_DEPTH-1, 3, MAX_DEPTH);
    in[i]=node;
  }
  std::vector<ot::TreeNode> in_cpy=in;
  std::vector<ot::TreeNode> out;

#ifdef __VERIFY__
  //Save input to file
  stringstream fname_;
  fname_<<"tmp/input_"<<myrank<<'\0';
  string fname=fname_.str();
  FILE* f=fopen(&fname[0],"wb+");
  fwrite(&in[0],N,sizeof(int),f);
  fclose(f);
#endif

  std::vector<double> tt(3);
  par::sampleSort<ot::TreeNode>(in, out, comm);
  in=in_cpy;
  par::bitonicSort<ot::TreeNode>(in, comm); out=in;
  in=in_cpy;
  par::sampleSort1<ot::TreeNode>(in, out, comm);
  in=in_cpy;

  //Sort
  double wtime;
  MPI_Barrier(comm);
  wtime=-omp_get_wtime();
  par::sampleSort<ot::TreeNode>(in, out, comm);
  MPI_Barrier(comm);
  wtime+=omp_get_wtime();
  //if(!myrank) std::cout<<N<<' '<<wtime<<'\n';
  tt[0]=wtime;
  in=in_cpy;

  MPI_Barrier(comm);
  wtime=-omp_get_wtime();
  par::bitonicSort<ot::TreeNode>(in, comm); out=in;
  MPI_Barrier(comm);
  wtime+=omp_get_wtime();
  //if(!myrank) std::cout<<N<<' '<<wtime<<'\n';
  tt[1]=wtime;
  in=in_cpy;

  MPI_Barrier(comm);
  wtime=-omp_get_wtime();
  par::sampleSort1<ot::TreeNode>(in, out, comm);
  MPI_Barrier(comm);
  wtime+=omp_get_wtime();
  //if(!myrank) std::cout<<N<<' '<<wtime<<'\n';
  tt[2]=wtime;
  in=in_cpy;

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

  return tt;

}

int main(int argc, char **argv){

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

  int j=0;
  MPI_Comm comm;
  for(int i=p;myrank<i && i>0 ;i=i>>1) j++;
  MPI_Comm_split(MPI_COMM_WORLD, j, myrank, &comm);

  std::vector<double> tt(3000000,0);
  long N_total = atol(argv[2]); //4000000;
  long N=100; //N_total/p;
  for(int k=0;k<7;k++){
    {
      std::vector<double> ttt=time_sort(N,MPI_COMM_WORLD);
      if(!myrank){
        tt[0*1000000+100*k+0]=ttt[0];
        tt[1*1000000+100*k+0]=ttt[1];
        tt[2*1000000+100*k+0]=ttt[2];
      }
    }
    {
      int myrank_;
      MPI_Comm_rank(comm, &myrank_);
      std::vector<double> ttt=time_sort(N,comm);
      if(!myrank_){
        tt[0*1000000+100*k+j]=ttt[0];
        tt[1*1000000+100*k+j]=ttt[1];
        tt[2*1000000+100*k+j]=ttt[2];
      }
    }
    N=N*4;
  }

  std::vector<double> tt_glb(3000000);
  MPI_Reduce(&tt[0], &tt_glb[0], 3000000, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if(!myrank){
    std::cout<<"\n\nNew Sort:\n";
    for(int i=0;i<j;i++){
      //int np=1u<<(j-i-1);
      int np=p;
      if(i>0) np=(p>>(i-1))-(p>>i);
      std::cout<<"P="<<np<<' ';
      for(int k=0;k<7;k++)
        std::cout<<tt_glb[0*1000000+100*k+i]<<' ';
      std::cout<<'\n';
    }
    std::cout<<"\n\nBitonic Sort:\n";
    for(int i=0;i<j;i++){
      int np=p;
      if(i>0) np=(p>>(i-1))-(p>>i);
      std::cout<<"P="<<np<<' ';
      for(int k=0;k<7;k++)
        std::cout<<tt_glb[1*1000000+100*k+i]<<' ';
      std::cout<<'\n';
    }
    std::cout<<"\n\nOld Sort:\n";
    for(int i=0;i<j;i++){
      int np=p;
      if(i>0) np=(p>>(i-1))-(p>>i);
      std::cout<<"P="<<np<<' ';
      for(int k=0;k<7;k++)
        std::cout<<tt_glb[2*1000000+100*k+i]<<' ';
      std::cout<<'\n';
    }
    std::cout<<"\n\n";
  }

  // Shut down MPI 
  MPI_Finalize();
  return 0;

}


