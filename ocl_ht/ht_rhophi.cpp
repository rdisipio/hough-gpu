//
//  ht_helix.cpp
//  
//
//  Created by Lorenzo Rinaldi on 29/04/14.
//  Adapted for OpenCL by Salavtore A. Tupputi on 25/08/2014
//  Compile with g++ -I /usr/local/cuda-6.0/targets/x86_64-linux/include -L /usr/local/cuda-6.0/targets/x86_64-linux/lib -o ht_rhophi ht_rhophi.cpp -lOpenCL
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>

#include <assert.h>
#include <sys/time.h>
#include <CL/cl.h>

using namespace std;

#define NHMAX 300
#define Nsec 4 // Numero settori in piano trasverso
#define Ntheta 16 // Numero settori in piano longitudinale
#define Nphi 1024 // Numero bin angolo polare
#define Nrho 1024 // Numero bin distanza radiale

#define rhomin 500.f // mm
#define rhomax 100000.f // mm
#define phimin 0.f // rad
#define phimax 2*M_PI // rad
#define thetamin 0.f // rad
#define thetamax M_PI // rad

#define ac_soglia 4 // soglia nella matrice di accumulazione

#define VERBOSE_DUMP
//#define DUMP_INFO
//#define PARALLEL_REDUX_MAX
//#define HOST_XCHECK

#define MALLOCHOST_OUTPUT_OUT
#define max_tracks_out 100

int acc_Mat [ Nsec ][ Ntheta ][ Nphi ] [ Nrho ];

float dtheta= M_PI/Ntheta;
float drho= (rhomax-rhomin)/Nrho;
float dphi= (phimax-phimin)/Nphi;

vector<float> x_values;
vector<float> y_values;
vector<float> z_values;


#ifndef PARALLEL_REDUX_MAX

struct track_param{
  int acc;
};

#ifndef MALLOCHOST_OUTPUT
struct track_param host_out_tracks[ Nsec * Ntheta * Nphi * Nrho ];
#endif

#endif

void read_inputFile(string file_path, unsigned int num_hits);
double wtime();

#define MAX_SOURCE_SIZE (0x100000)

void help(char* prog) {
  
  printf("Use %s [-l #loops] [-n #hitsToRead] [-h] \n\n", prog);
  printf("  -l loops        Number of executions (Default: 1).\n");
  printf("  -n hits         Number of hits to read from input file (Default: 236).\n");
  printf("  -f input file   Name for input file (Default: hits-5000.txt).\n");
  printf("  -h              This help.\n");

}

void fetchKernel(char const * fileName, char * & source_str, size_t& source_size){
  
  FILE *fp;
  fp = fopen(fileName, "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel in %s.\n", fileName);
    exit(1);
  }
  
  source_str = (char *)malloc((size_t) MAX_SOURCE_SIZE);
  assert(source_str);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);
  
}

void printDInfo(const char * info, long unsigned * val, unsigned int size = 1){
  
  cout << "== DEVICE INFO == " << info << " ... ";
  if (size == 1)
    cout << *((unsigned long *)val) << endl;
  else {
    cout << "[";
    for (unsigned int i=0; i<size; i++)
      cout << " " << val[i];
    cout << " ]" << endl;
  }
  
}

void CL_CALLBACK context_notify(const char *notify_message, const void *private_info, size_t cb, void *user_data)
{
  printf("Notification:\n\t%s\n\t%s\n", notify_message, private_info);
}

void CL_CALLBACK callBack(cl_event evnt,cl_int cmd_sts,void *user_data)
{

  printf("CALLBACK! CALLBACK!\n");

}

int main(int argc, char* argv[]){
  
  double t_start = wtime();

  unsigned int N_LOOPS = 1;
  unsigned int N_HITS = 236;
  string FILENAME = "hits-5000.txt";
  int c;

  //getting command line options
  while ( (c = getopt(argc, argv, "l:n:f:h")) != -1 ) {
    switch(c) {
      
    case 'n':
      N_HITS = atoi(optarg);
      break;
    case 'l':
      N_LOOPS = atoi(optarg);
      break;
    case 'f':
      FILENAME = optarg;
      break;
    case 'h':
      help(argv[0]);
      return 0;
      break;
    default:
      printf("Unkown option!\n");
      help(argv[0]);
      return 0;
    }
  }

  
  cl_platform_id platform_id = NULL;
  cl_device_id  device_id = NULL;;
  cl_device_info device_info = NULL;
  cl_context context = NULL;
  cl_command_queue command_queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel[3] = { NULL, NULL, NULL };
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret,ret1,ret2;
  size_t maxWGS;

  //info su piattaforma e device, contesto e command queue sono oggetti comuni a tutti i kernel
  double t_init = wtime();
  ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);  
  ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

  double t_info = wtime();
  ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWGS, NULL);

#ifdef DUMP_INFO
  cl_int int_sz, float_sz, ncores, fclock;
  cl_uint glocchline_sz;
  cl_ulong locmem_sz, glocch_sz, glomem_sz;
  cl_device_local_mem_type locmem_tp;
  cl_device_mem_cache_type gcchmem_tp;
  size_t t_resol;

  // --- Variabile per numero della GPU per la divisione del lavoro appropriata
  unsigned int maxWID;
  unsigned long maxAllocSize;

  // Get Platform/Device Information
  printDInfo("CL_DEVICE_MAX_WORK_GROUP_SIZE",&maxWGS);
  ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(unsigned int), &maxWID, NULL);
  printDInfo("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS",(long unsigned *) &maxWID);
  ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(unsigned long), &maxAllocSize, NULL);
  printDInfo("CL_DEVICE_MAX_MEM_ALLOC_SIZE",&maxAllocSize);
  ret = clGetDeviceInfo(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(cl_uint), &int_sz, NULL);
  printDInfo("CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT",(long unsigned *) &int_sz);
  ret = clGetDeviceInfo(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(cl_uint), &float_sz, NULL);
  printDInfo("CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT",(long unsigned *) &float_sz);
  ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &ncores, NULL);
  printDInfo("CL_DEVICE_MAX_COMPUTE_UNITS",(long unsigned *) &ncores);
  ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &fclock, NULL);
  printDInfo("CL_DEVICE_MAX_CLOCK_FREQUENCY",(long unsigned *) &fclock);
  ret = clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &locmem_sz, NULL);
  printDInfo("CL_DEVICE_LOCAL_MEM_SIZE",&locmem_sz);
  ret = clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(locmem_tp), &locmem_tp, NULL);
  printDInfo("CL_DEVICE_LOCAL_MEM_TYPE",(long unsigned *) &locmem_tp);
  ret = clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &glomem_sz, NULL);
  printDInfo("CL_DEVICE_GLOBAL_MEM_SIZE",&glomem_sz);
  ret = clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, sizeof(gcchmem_tp), &gcchmem_tp, NULL);
  printDInfo("CL_DEVICE_GLOBAL_MEM_CACHE_TYPE",(long unsigned *) &gcchmem_tp);
  ret = clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &glocch_sz, NULL);
  printDInfo("CL_DEVICE_GLOBAL_MEM_CACHE_SIZE",&glocch_sz);
  ret = clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cl_uint), &glocchline_sz, NULL);
  printDInfo("CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE",(long unsigned *) &glocchline_sz);
  size_t maxItems[maxWID];
  ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxItems), maxItems, NULL);
  printDInfo("CL_DEVICE_MAX_WORK_ITEM_SIZES",maxItems,maxWID);
  ret = clGetDeviceInfo(device_id, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(t_resol), &t_resol, NULL);
  printDInfo("CL_DEVICE_PROFILING_TIMER_RESOLUTION",(long unsigned *)&t_resol);
  cout << "DEVICE INFO TIME " << wtime() - t_info << endl;
#endif

  //Create OpenCL Context
  context = clCreateContext(NULL, 1, &device_id, context_notify, NULL, &ret);

  //Create command queue 
  command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
  t_init = wtime() - t_init;
  //cout << "OCL INIT TIME " << wtime() - t_init <<  endl;

  // Load kernel(s) from source file
  char fileName[]="ht_rhophi.cl";
  char * krnStr;
  size_t krnSize;
  fetchKernel(fileName, krnStr, krnSize);
  //cout<< "kernel fetched " << krnSize << " "<< krnStr[0] << krnStr[1] << krnStr[2] << endl;

  // Create kernel program from source file
  double t_prog = wtime();
  program = clCreateProgramWithSource(context, 1, (const char **)&krnStr, (const size_t *)&krnSize, &ret);
  //cout << "program creation exit status" << ret << " within " << wtime() - t_prog << endl;

  // Build program
  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  //cout << "build program " << ret << " " << wtime() - t_prog << endl;
  char buildLog[10240];
  clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
  if (buildLog && buildLog[0] == '\n')
    cout << "Kernels successfully built!" << endl;
  else
    cout << "buildLog..." << buildLog << endl;
  //cout << "BUILD TIME " << wtime() - t_prog<< endl;

  // Create data parallel OpenCL kernel
  char kernelNames[4][16] = { "memset_int", "voteHoughSpace", "memset_int2", "findRelativeMax" };
  kernel[0] = clCreateKernel(program, kernelNames[0], &ret);
  kernel[1] = clCreateKernel(program, kernelNames[1], &ret1);
  kernel[2] = clCreateKernel(program, kernelNames[2], &ret2);
  kernel[3] = clCreateKernel(program, kernelNames[3], &ret2);
  cout << "create kernel " << ret << "&" << ret1 << "&" << ret2 << "&" << ret2 << " " << wtime() - t_prog << endl;
  
#ifndef DUMP_KINFO
  
  for (unsigned int kk=0; kk<4; kk++){
    size_t k_WGS = NULL, k_WGS_mult = NULL, k_comp_WGS[3];
    cl_ulong k_localMSize = NULL, k_privMSize = NULL;
    ret = clGetKernelWorkGroupInfo(kernel[kk], device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(k_WGS), &k_WGS, NULL);
    ret = clGetKernelWorkGroupInfo(kernel[kk], device_id, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(size_t) * 3, &k_comp_WGS, NULL);
    ret = clGetKernelWorkGroupInfo(kernel[kk], device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(k_WGS_mult), &k_WGS_mult, NULL);
    ret = clGetKernelWorkGroupInfo(kernel[kk], device_id, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(cl_ulong), &k_localMSize, NULL);
    ret = clGetKernelWorkGroupInfo(kernel[kk], device_id, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(cl_ulong), &k_privMSize, NULL);
    cout << "INFO FOR KERNEL " << kernelNames[kk] << endl;
    printDInfo("CL_KERNEL_WORK_GROUP_SIZE",(long unsigned *) &k_WGS);
    printDInfo("CL_KERNEL_COMPILE_WORK_GROUP_SIZE",(long unsigned *) &k_comp_WGS, 3);
    printDInfo("CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE",(long unsigned *) &k_WGS_mult);
    printDInfo("CL_KERNEL_LOCAL_MEM_SIZE", &k_localMSize);
    printDInfo("CL_KERNEL_PRIVATE_MEM_SIZE", &k_privMSize);
  }

#endif

  cl_mem xBuf = NULL, yBuf = NULL, zBuf = NULL;
  cl_mem hostPinAMBuf = NULL;
  cl_mem devAMBuf = NULL;

  float *x_values_temp = NULL;
  float *y_values_temp = NULL;
  float *z_values_temp = NULL;
  int *krnAccMat = NULL;
  
  //riempi i valori dentro x_values , y_values , z_values
  read_inputFile("hits-5000.txt", N_HITS);
  
  double t_0 = wtime(), t_1, t_2, t_3, t_4;
  x_values_temp = (float*) malloc(sizeof(float) * x_values.size());
  y_values_temp = (float*) malloc(sizeof(float) * y_values.size());
  z_values_temp = (float*) malloc(sizeof(float) * z_values.size());

  for(unsigned int i = 0; i < x_values.size(); i++){
    x_values_temp[i] = x_values.at(i);
    y_values_temp[i] = y_values.at(i);
    z_values_temp[i] = z_values.at(i);
  }
  t_0 = wtime() - t_0;

  //Create data buffers for "helper" and voteHoughSpaceKernel
  double t_buf_xyz = wtime();
  xBuf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * x_values.size(), NULL, &ret);
  yBuf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * y_values.size(), NULL, &ret);
  zBuf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * z_values.size(), NULL, &ret);
  t_buf_xyz = wtime() - t_buf_xyz;
  t_1 = wtime();
  devAMBuf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * (Nsec*Ntheta*Nphi*Nrho), NULL, &ret);
  hostPinAMBuf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(int) * (Nsec*Ntheta*Nphi*Nrho), NULL, &ret);
  t_1 = wtime() - t_1;

  cl_event writeMapEvents[6];

  cl_event startWrite = clCreateUserEvent(context, &ret1);
  ret = clEnqueueWriteBuffer(command_queue, xBuf, CL_FALSE, 0, sizeof(float) * x_values.size(), x_values_temp, 1, &startWrite, &(writeMapEvents[0]));
  ret |= clEnqueueWriteBuffer(command_queue, yBuf, CL_FALSE, 0, sizeof(float) * y_values.size(), y_values_temp, 1, &startWrite, &(writeMapEvents[1]));
  ret |= clEnqueueWriteBuffer(command_queue, zBuf, CL_FALSE, 0, sizeof(float) * z_values.size(), z_values_temp, 1, &startWrite, &(writeMapEvents[2]));
  //map host pinned memory buffer
  krnAccMat = (int *) clEnqueueMapBuffer(command_queue, hostPinAMBuf, CL_FALSE, CL_MAP_READ, 0, sizeof(int) * (Nsec*Ntheta*Nphi*Nrho), 1, &startWrite, &(writeMapEvents[3]), &ret);

#ifndef PARALLEL_REDUX_MAX
  
  unsigned int * nMaxima = (unsigned int *) calloc(1, sizeof(*nMaxima));
  
  //Create data buffers for findRelativeMax kernel
  cl_mem nMaxBuf = NULL;
  cl_mem devIdOBuf = NULL;
  cl_mem hostPinIdOBuf = NULL;
  struct track_param * krnIdxOut = NULL;
  
  double t_maxes = wtime();
  nMaxBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned int), NULL, &ret);
  devIdOBuf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(struct track_param) * (Nsec*Ntheta*Nphi*Nrho), NULL, &ret);
  hostPinIdOBuf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(struct track_param) * (Nsec*Ntheta*Nphi*Nrho), NULL, &ret);
  t_maxes = wtime() - t_maxes;

  ret = clEnqueueWriteBuffer(command_queue, nMaxBuf, CL_FALSE, 0, sizeof(unsigned int), nMaxima, 1, &startWrite, &(writeMapEvents[4]));
  krnIdxOut = (struct track_param *) clEnqueueMapBuffer(command_queue, hostPinIdOBuf, CL_FALSE, CL_MAP_READ, 0, sizeof(struct track_param) * (Nsec*Ntheta*Nphi*Nrho), 1, &startWrite, &(writeMapEvents[5]), &ret);

  unsigned int nEvs = 6;  
#else
  unsigned int nEvs = 4;
#endif
    
  cl_ulong w1end[nEvs+1], w1start[nEvs+1];
  w1end[nEvs] = 0; w1start[nEvs] = 0xFFFFFFFFFFFFFFFF;

  double t_write1 = wtime();
  clSetUserEventStatus(startWrite, CL_COMPLETE);
  ret = clWaitForEvents(nEvs, writeMapEvents);
  t_write1 = wtime() - t_write1;

  for (unsigned int ev = 0; ev < nEvs; ev++){
    ret = clGetEventProfilingInfo(writeMapEvents[ev], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &(w1end[ev]), 0);
    ret |= clGetEventProfilingInfo(writeMapEvents[ev], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &(w1start[ev]), 0);
    cout << ev << " " << w1start[ev] << " " << w1end[ev] << " " << (w1end[ev] - w1start[ev]) * 1.0e-6f << " ms" << endl;
    if (w1start[ev] < w1start[nEvs])
      w1start[nEvs] = w1start[ev];
    if (w1end[ev] > w1end[nEvs])
      w1end[nEvs] = w1end[ev];
  }
  cout << "Writing/Mapping for buffers in OCL time: " << (w1end[nEvs] - w1start[nEvs]) * 1.0e-6f << " ms (or " << t_write1 << " s)" << endl;
  
  t_0 += t_buf_xyz;
  for (unsigned int b0=0; b0<3; b0++)
    t_0 += (w1end[b0] - w1start[b0])*1.0e-6f;
  cout << "Input malloc and copy HtoD time: " << t_0 << " ms" << endl;

  t_1 += t_maxes;
  for (unsigned int b1=3; b1<nEvs; b1++)
    t_1 += (w1end[b1] - w1start[b1])*1.0e-6f;

  double t_tot = (w1end[nEvs] - w1start[nEvs]) * 1.0e-6f;

  //executions loop
  for(unsigned int loop = 0; loop < N_LOOPS; loop++){
    
    //KERNEL_0 "helper" kernel initializing to 0 the accMat
    int val = 0;
    ret = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void *) &devAMBuf);
    ret = clSetKernelArg(kernel[0], 1, sizeof(int), (void *) &val);
    size_t const local_items [2] = { Nphi, maxWGS/Nphi };
    size_t const global_items [2] = { Nrho * Nphi, Ntheta * Nsec};//(maxWGS/Nphi)};

    cl_event krn1Event;
    double t_krn1 = wtime();
    ret = clEnqueueNDRangeKernel(command_queue, kernel[0], 2, NULL, global_items, local_items, 0, 0, &krn1Event);
    cout << "Kernel1 launched (exit=" << ret << ") with Global size [0]: " << global_items[0] << " [1]: " << global_items[1] <<"; Local size [0]: " << local_items[0] << " [1]: "<< local_items[1] << endl;

    cl_ulong end, start;
    ret = clWaitForEvents(1, &krn1Event);
    ret = clGetEventProfilingInfo(krn1Event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, 0);
    ret |= clGetEventProfilingInfo(krn1Event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, 0);
    cout << "Kernel1 OCL time: "<<(end-start)*1.0e-6f<<" ms (or "<< wtime() - t_krn1 << " s)" <<endl;
    clFinish(command_queue);

    if (N_LOOPS == 1){
      t_1 += (end-start)*1.0e-6f;
      cout << "malloc dev_accMat and memset(0) " << t_1 << " ms" << endl;
    }

    //KERNEL_1 voteHoughMatrix
    ret = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), (void *)&xBuf);
    ret |= clSetKernelArg(kernel[1], 1, sizeof(cl_mem), (void *)&yBuf);
    ret |= clSetKernelArg(kernel[1], 2, sizeof(cl_mem), (void *)&zBuf);
    ret |= clSetKernelArg(kernel[1], 3, sizeof(cl_mem), (void *)&devAMBuf);
    ret |= clSetKernelArg(kernel[1], 4, sizeof(float), 0);
    ret |= clSetKernelArg(kernel[1], 5, sizeof(float), 0);
    ret |= clSetKernelArg(kernel[1], 6, sizeof(float), 0);
    
    size_t local_size = (size_t) Nphi;
    size_t global_size = x_values.size() * local_size;

    cl_event krn2Event;
    double t_krn2 = wtime();
    ret = clEnqueueNDRangeKernel(command_queue, kernel[1], 1, NULL, &global_size, &local_size, 0, 0, &krn2Event);
    cout << "Kernel2 launched (exit=" << ret << ") with Global size: " << global_size << "; Local size: " << local_size << endl;

    ret = clWaitForEvents(1, &krn2Event);
    ret = clGetEventProfilingInfo(krn2Event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, 0);
    ret |= clGetEventProfilingInfo(krn2Event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, 0);
    t_2 = (end-start)*1.0e-6f;
    cout<<"Kernel2 OCL time: " << t_2 << " ms (or "<< wtime() - t_krn2 << " s)" <<endl;

#ifdef VERBOSE_DUMP
    double t_read1 = wtime();
    ret = clEnqueueReadBuffer(command_queue, devAMBuf, CL_TRUE, 0, sizeof(int) * (Nsec*Ntheta*Nphi*Nrho), krnAccMat, 0, NULL, NULL);
    cout << "Read devAccMat buffer... " <<ret << " " << wtime() - t_read1 << endl;
#endif

#ifdef HOST_XCHECK
    memset(&acc_Mat, 0, (sizeof(int)*(Nsec*Ntheta*Nphi*Nrho)) );
    cout << "Run VoteHoughSpace on host ... \n" ;
    
    double t_host1 = wtime();
    int aggr(0);
    for(unsigned int i = 0; i < x_values.size(); i++){
      
      float R2=x_values.at(i)*x_values.at(i)+y_values.at(i)*y_values.at(i);
      float theta=acos(z_values.at(i)/sqrt(R2+z_values.at(i)*z_values.at(i)));
      int ith=floor((theta/dtheta));
      //int ith=(int) (theta/dtheta)+0.5f;
      
      float sec=atan2(y_values.at(i),x_values.at(i));
      if (sec<0.f)
	sec=2*M_PI+sec;
      int isec=floor(sec/2/M_PI*Nsec);
      //int isec=int(sec/2/M_PI*Nsec);
      
      for(int iphi = 0; iphi < Nphi; iphi++){
	float phi=phimin+iphi*dphi;
	float rho=R2/2.f/(x_values.at(i)*cos(phi)+y_values.at(i)*sin(phi));
	int irho=floor((rho-rhomin)/drho);
	//int irho=(int)((rho-rhomin)/drho)+0.5f;
	if (rho<=rhomax && rho>rhomin){
	  acc_Mat[isec][ith][iphi][irho]++;
	  aggr++;
	}
      }
    }
    cout << "in " << wtime() - t_host1 << " s. aggr = " << aggr << endl;

#ifdef VERBOSE_DUMP
    //Xcheck voteHoughMatrix
    unsigned int corretto(0), errore(0), devzero(0), hostzero(0), letto(0), daggr(0);
    for(unsigned int isec = 0; isec < Nsec; isec++)
      for(unsigned int ith = 0; ith < Ntheta; ith++)
	for(unsigned int iphi = 0; iphi < Nphi; iphi++)
	  for(unsigned int irho = 0; irho < Nrho; irho++){
	    unsigned int idx = (isec*(Ntheta*Nphi*Nrho))+((ith*Nphi*Nrho) + ((iphi*Nrho)+irho));
	    if(acc_Mat[isec][ith][irho][iphi] != krnAccMat[idx]){
	      int delta = acc_Mat[isec][ith][irho][iphi] - krnAccMat[idx];
	      //printf("DELTA = %d!! acc_Mat[%d][%d][%d][%d] %d - krnAccMat[%d][%d][%d][%d] %d \n", delta, isec, ith, iphi, irho, acc_Mat[isec][ith][iphi][irho], isec, ith, iphi, irho, krnAccMat[idx]);
	      //printf("%d %d %d %d DELTA %d\n",isec,ith,iphi,irho,delta);
	      errore++;
	    }
	    else 
	      corretto++;
	    if (acc_Mat[isec][ith][irho][iphi] != 0)
	      hostzero++;
	    if ( krnAccMat[idx] != 0){
	      daggr += krnAccMat[idx];
	      devzero++;
	    }
	    letto++;
	  }
	  cout << "corretti " << corretto << " sbagliati " << errore << " letti " << letto << "(" << (float)errore/(float)letto << "%) DEVnonnulli " << devzero << " HOSTnonulli " << hostzero << " DEVAGGR " << daggr << endl;
#endif

#endif
    
    //trova il massimo relativo
    unsigned int host_NMrel = 0;
    
#ifndef PARALLEL_REDUX_MAX

    //HELPER KERNEL_3
    int value = -1;
    ret = clSetKernelArg(kernel[2], 0, sizeof(cl_mem), (void *) &devIdOBuf);
    ret |= clSetKernelArg(kernel[2], 1, sizeof(int), (void *) &value);
    size_t const loc_items [2] = { Nphi, maxWGS/Nphi };
    size_t const glob_items [2] = { Nrho * Nphi, Ntheta * Nsec};//(maxWGS/Nphi)};

    cl_event krn3Event;
    double t_krn3 = wtime();
    ret = clEnqueueNDRangeKernel(command_queue, kernel[2], 2, NULL, glob_items, loc_items, 0, 0, &krn3Event);
    cout << "Kernel3 launched (exit=" << ret << ") with Global size [0]: " << glob_items[0] << " [1]: " << glob_items[1] <<"; Local size [0]: " << loc_items[0] << " [1]: "<< loc_items[1] << endl;

    ret = clWaitForEvents(1, &krn3Event);
    ret = clGetEventProfilingInfo(krn3Event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, 0);
    ret |= clGetEventProfilingInfo(krn3Event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, 0);
    cout << "Kernel3 OCL time: "<<(end-start)*1.0e-6f<<" ms (or "<< wtime() - t_krn3 << " s)" <<endl;
    clFinish(command_queue);

    if (N_LOOPS == 1){
      t_1 += (end-start)*1.0e-6f;
      cout << "malloc dev_accMat and memset(0) for both matrixes " << t_1 << " ms" << endl;
    }

    //KERNEL_4 findRelativMax
    //dividiamo il lavoro in base al numero massimo di thread disponibili in un singolo thread-block
    unsigned int dim_x_block = Nphi;
    unsigned int dim_y_block = maxWGS/dim_x_block;
    unsigned int dim_x_grid = Nrho * dim_x_block;//Nsec * dim_x_block;
    unsigned int dim_y_grid = Ntheta * Nsec;//dim_y_block;//(Nrho/dim_y_block);

    size_t const local_item_size [2] = { dim_x_block, dim_y_block };
    size_t const global_item_size [2] = { dim_x_grid, dim_y_grid };

    ret = clSetKernelArg(kernel[3], 0, sizeof(cl_mem), (void *)&devAMBuf);
    ret |= clSetKernelArg(kernel[3], 1, sizeof(cl_mem), (void *)&devIdOBuf);
    ret |= clSetKernelArg(kernel[3], 2, sizeof(cl_mem), (void *)&nMaxBuf);
    cout << "Kernel3 arguments set " << ret <<endl;

    clFinish(command_queue);
      
    cl_event krn4Event;
    double t_krn4 = wtime();
    ret = clEnqueueNDRangeKernel(command_queue, kernel[3], 2, NULL, global_item_size, local_item_size,  0, 0, &krn4Event);
    //fprintf(stderr, "line %d: err %d\n", __LINE__, ret);
    cout << "Kernel4 launched (exit=" << ret << ") with Global size [0]: " <<dim_x_grid << " [1]: " << dim_y_grid << "; Local size [0]: " << dim_x_block << " [1]: " << dim_y_block << endl;

    //ret = clSetEventCallback(krn3Event, CL_COMPLETE, &callBack, NULL);

    ret = clWaitForEvents(1, &krn4Event);
    ret = clGetEventProfilingInfo(krn4Event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, 0);
    ret |= clGetEventProfilingInfo(krn4Event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, 0);
    t_3 = (end-start)*1.0e-6f;
    cout<<"Kernel4 OCL time: " << t_3 << " ms (or "<< wtime() - t_krn4 << " s)" <<endl;
    clFinish(command_queue);
    
    cl_event cpyResult[2];
    cl_event startRead = clCreateUserEvent(context, &ret1);
#ifdef MALLOCHOST_OUTPUT
    ret = clEnqueueReadBuffer(command_queue, devIdOBuf, CL_FALSE, 0, sizeof(struct track_param) * (Nsec * Ntheta * Nphi * Nrho), krnIdxOut, 0, 0, &(cpyResult[0]));
    cout << "read devIdOBuf inside MALLOCHOST_OUTPUT " << ret <<endl;
#else
    ret = clEnqueueReadBuffer(command_queue, devIdOBuf, CL_FALSE, 0, sizeof(struct track_param) * (Nsec * Ntheta * Nphi * Nrho), krnIdxOut, 0, 0, &(cpyResult[0]));
    //fprintf(stderr, "line %d: err %d\n", __LINE__, ret);
    cout << "read devIdOBuf outside MALLOCHOST_OUTPUT " << ret <<endl;
#endif
    ret = clEnqueueReadBuffer(command_queue, nMaxBuf, CL_FALSE, 0, sizeof(int), nMaxima, 0, 0, &(cpyResult[1]));

    cl_ulong r1end[3], r1start[3];
    r1end[2] = 0; r1start[2] = 0xFFFFFFFFFFFFFFFF;

    double t_res = wtime();
    clSetUserEventStatus(startRead, CL_COMPLETE);
    ret = clWaitForEvents(2, cpyResult);
    t_res = wtime() - t_res;

  for (unsigned int ev = 0; ev < 2; ev++){
    ret = clGetEventProfilingInfo(cpyResult[ev], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &(r1end[ev]), 0);
    ret |= clGetEventProfilingInfo(cpyResult[ev], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &(r1start[ev]), 0);
    cout << ev << " " << r1start[ev] << " " << r1end[ev] << " " << (r1end[ev] - r1start[ev]) * 1.0e-6f << " ms" << endl;
    if (r1start[ev] < r1start[2])
      r1start[2] = r1start[ev];
    if (r1end[ev] > r1end[2])
      r1end[2] = r1end[ev];
  }
  t_4 = (r1end[2] - r1start[2]) * 1.0e-6f;
  cout << "Reading buffers in OCL time: " << t_4 << " ms (or " << t_res << " s)" << endl;

#ifdef VERBOSE_DUMP
    cout << "Number of relative maxima from GPU "<< *nMaxima ;

    unsigned int ntracks = 0;
    int acc_ave = 0;
      
    for(unsigned int i = 0; ((i < (Nsec * Ntheta * Nphi * Nrho)) && (ntracks < *nMaxima)); i++){
	
      if(krnIdxOut[i].acc > -1){
	//cout << "track " << ntracks << " acc value = " << krnIdxOut[i].acc << " [" << i << "]" << endl;
	ntracks++;
	acc_ave += krnIdxOut[i].acc;
      }
    }
    cout << "; Tracks found " << ntracks << " with average acceptance value of " << (float)acc_ave/(float)ntracks << endl;
#endif
    //free mem
    //ret = clReleaseMemObject(nMaxBuf);
    //cout << "release nMaxBuf " <<ret <<endl;
    ret = clReleaseMemObject(devIdOBuf);
    ret = clReleaseMemObject(hostPinIdOBuf);
    cout << "release idxOutBuf " <<ret <<endl;

#endif

#ifdef HOST_XCHECK

    //free(krnIdxOut);
    cout << "Run FindRelativeMax on host ... ";
    host_NMrel = 0;
      
    int accumax = -1;
    int iphiMax = 0;
    int irhoMax = 0;
    int ithMax = 0;
    int isecMax = 0;
    int ave_acc = 0;

    double t_host2 = wtime();      
    for(unsigned int isec = 0; isec < Nsec; isec++){
      for(unsigned int ith = 0; ith < Ntheta; ith++){
	for(unsigned int iphi = 1; iphi < Nphi-1; iphi++){
	  for(unsigned int irho = 1; irho < Nrho-1; irho++){
		      
	    float acc=acc_Mat[isec][ith][irho][iphi];
	    if (acc >= ac_soglia){
	      if (acc > accumax)
		accumax=acc;
			  
	      if(acc > acc_Mat[isec][ith][irho][iphi-1] && acc >= acc_Mat[isec][ith][irho][iphi+1]){
		if(acc > acc_Mat[isec][ith][irho-1][iphi] && acc >= acc_Mat[isec][ith][irho+1][iphi]){
		  accumax = acc_Mat[isec][ith][irho][iphi];
		  ave_acc += acc;
		  host_NMrel++;
		  ithMax=ith;
		  irhoMax=irho;
		  iphiMax=iphi;
		  isecMax=isec;
		  float t_th=(thetamin+ithMax*dtheta)*360.f/M_PI;
		  float t_rho=rhomin+irhoMax*drho;
		  float t_phi=phimin+iphiMax*dphi;

		}
	      }
	    }
	  }
	}
      }
    }

#ifdef VERBOSE_DUMP
    cout << " in " << wtime() - t_host2 << "; NMrel from CPU "<< host_NMrel << " with " << (float)ave_acc/(float)host_NMrel << " acceptance average" <<endl;
#endif

#endif

    ret = clReleaseMemObject(devAMBuf);
    cout << "release devAMBuf " << ret <<endl;      

  }
  
  // #ifdef MALLOCHOST_OUTPUT      
  //   ret = clReleaseMemObject(trksmobj);
  //   cout << "release trksmobj " << ret <<endl;
  // #endif
  
  ret = clReleaseMemObject(xBuf);
  cout << "release xmobj " <<ret <<endl;
  ret = clReleaseMemObject(yBuf);
  cout << "release ymobj " <<ret <<endl;
  ret = clReleaseMemObject(zBuf);
  cout << "release zmobj " <<ret <<endl;
    
  free(x_values_temp);
  free(y_values_temp);
  free(z_values_temp);
    
  x_values.clear();
  y_values.clear();
  z_values.clear();
    
  // ret = clFlush(command_queue);
  // cout << ret << endl;
  // ret = clFinish(command_queue);
  // cout << ret << endl;
  // ret = clReleaseMemObject(devAMBuf);
  // cout << ret << endl;
  ret = clReleaseKernel(*kernel);
  cout << ret << endl;
  ret = clReleaseProgram(program);
  cout << ret << endl;
  ret = clReleaseCommandQueue(command_queue);
  cout << ret << endl;
  ret = clReleaseContext(context);
  cout << "released everything else " << ret << endl;
 
  free(krnStr);
  cout << "HITS" << N_HITS << " " << t_0 << " " << t_1 << " " << t_2 << " " << t_3 << " " << t_4 << " " << t_tot << endl;;
  cout << wtime() - t_start << " sec: breve ma intenso" << endl;
  return 0;
}


/*****************************
 * file opener
 *****************************/

void read_inputFile(string file_path, unsigned int num_hits)
{
    
  ifstream input_f;
    
  string line;
  string value;
    
  stringstream ss;
  unsigned int val_iter;
    
  unsigned int line_read = 0;
    
  input_f.open(file_path.c_str());
  if (input_f.is_open())
    {
      while ( getline (input_f,line) && (line_read < num_hits))
        {
	  val_iter = 0;
	  ss.str(line);
	  //prendiamo dati direttamente dal file ASCII in input
	  while(ss >> value){
	    //i valori che ci interessano sono X, Y e Z
	    if (val_iter == 0) x_values.push_back(atof(value.c_str()));
	    else if (val_iter == 1) y_values.push_back(atof(value.c_str()));
	    else if (val_iter == 2) z_values.push_back(atof(value.c_str()));
	    val_iter++;
                
	  }
	  ss.clear();
	  line_read++;
	  //if (line_read ==  MAXHITS)
	  // break;
        }
      input_f.close();
    }
  else
    fprintf(stderr,"file not found!");

}

double wtime()
{
  /* Use a generic timer */
  static int sec = -1;
  struct timeval tv;
  gettimeofday(&tv, NULL);
  if (sec < 0) sec = tv.tv_sec;
  return (tv.tv_sec - sec) + 1.0e-6*tv.tv_usec;
}
