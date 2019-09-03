#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#define NHMAX 300
#define Nsec 4
#define Ntheta 16
#define Nphi 1024
#define Nrho 1024

#define rhomin 500.f // mm
#define rhomax 100000.f // mm
#define phimin 0.f // rad
#define phimax 2*M_PI // rad
#define thetamin 0.f // rad
#define thetamax M_PI // rad

#define ac_soglia 4 // soglia nella matrice di accumulazione

__constant float dtheta= M_PI/Ntheta;
__constant float drho= (rhomax-rhomin)/Nrho;
__constant float dphi= (phimax-phimin)/Nphi;

#define get4DIndex(s,t,p,r) ((s)*(Ntheta*Nphi*Nrho))+(((t)*Nphi*Nrho) +(((r)*Nphi)+(p)))

__kernel void memset_int(__global int *dev_accMat, __private int val) { 
  
  unsigned int isec = get_group_id(1) % Nsec;
  unsigned int ith = get_group_id(1)/Nsec;
  unsigned int irho = get_local_id(0);
  unsigned int iphi = get_group_id(0);

  dev_accMat[get4DIndex(isec, ith, irho, iphi)] = val;

}


__kernel void voteHoughSpace(__global float *dev_x_values, __global float *dev_y_values, __global float *dev_z_values, __global int *dev_accMat, __local float *x_val, __local float *y_val, __local float *z_val){
   
  //float x_val, y_val, z_val;
  if(get_local_id(0) == 0){
    x_val[0] = dev_x_values[get_group_id(0)];
    y_val[0] = dev_y_values[get_group_id(0)];
    z_val[0] = dev_z_values[get_group_id(0)];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  
  int iphi = get_local_id(0);

  float R2 = x_val[0]*x_val[0] + y_val[0]*y_val[0];
  float theta=acos(z_val[0]/sqrt(R2+z_val[0]*z_val[0]));
  
  int ith=(int) (theta/dtheta)+0.5f;
  
  float sec=atan2(y_val[0],x_val[0]);
  if (sec<0.f)
  {
    sec=2*M_PI+sec;
  }
  int isec=(int) (sec/2/M_PI*Nsec);
  
  float phi=phimin+iphi*dphi;
  float rho=R2/2.f/(x_val[0]*cos(phi)+y_val[0]*sin(phi));
  int irho=(int)((rho-rhomin)/drho)+0.5f;
  
  //int accu_index = (isec*(Ntheta*Nphi*Nrho))+((ith*Nphi*Nrho) +((irho*Nphi)+iphi));
  int accu_index = get4DIndex(isec, ith, iphi, irho);  

  if (rho<=rhomax && rho>rhomin)
  {
    atom_inc(&(dev_accMat[accu_index]));
  }
}

#ifndef PARALLEL_REDUX_MAX

struct track_param{
      int acc;
    };

__kernel void memset_int2(__global struct track_param *dev_output, __private int val) { 
  
  unsigned int isec = get_group_id(1) % Nsec;
  unsigned int ith = get_group_id(1)/Nsec;
  unsigned int irho = get_local_id(0);
  unsigned int iphi = get_group_id(0);

  dev_output[get4DIndex(isec, ith, irho, iphi)].acc = val;


}

__kernel void findRelativeMax(__global int *dev_accMat, __global struct track_param *dev_output, __global unsigned int * NMrel){//, __local unsigned int *irho){

  unsigned int isec = get_group_id(1) % Nsec;
  unsigned int ith = get_group_id(1)/Nsec;//(int) get_group_id(1) / (Ntheta/get_local_size(1));
  unsigned int irho = get_local_id(0);//get_group_id(1) % (Nrho/get_local_size(1));
  unsigned int iphi = get_group_id(0);

  if((iphi > 0) && (irho > 0) && (iphi < Nphi-1) && (irho < Nrho-1)){
    
    int globalIndex = get4DIndex(isec, ith, irho, iphi);

    //each thread is assigned to one point of the accum. matrix:
    int acc = dev_accMat[globalIndex];

    if (acc >= ac_soglia){
      
      //if(acc > dev_accMat[globalIndex-Nrho] && acc >= dev_accMat[globalIndex+Nrho]){
        //if(acc > dev_accMat[globalIndex-1] && acc >= dev_accMat[globalIndex+1]){
      if(acc > dev_accMat[get4DIndex(isec, ith, irho-1, iphi)] && acc >= dev_accMat[get4DIndex(isec, ith, irho+1, iphi)]){
        if(acc > dev_accMat[get4DIndex(isec, ith, irho, iphi-1)] && acc >= dev_accMat[get4DIndex(isec, ith, irho, iphi+1)]){
      
          atom_inc(NMrel);
          dev_output[globalIndex].acc = acc;

	}
      }
    }
  }
}

#else

__kernel void reduceParallelMax(__global int *dev_accMat, __global int *dev_output, __global int *dev_maxRelOutput, unsigned int N, __local int *sdata){
  
  
  int* max_sdata = (int *) sdata;

  unsigned int loc_dim = get_local_size(0);
  int* relMax_sdata = (int *) &sdata[loc_dim];
  
  unsigned int tid = get_local_id(0);
  unsigned int blockId = get_group_id(1) * loc_dim + get_group_id(0);
  unsigned int i = blockId * loc_dim + get_local_id(0);
  
  if(i < N){
  
    max_sdata[tid] = dev_accMat[i];
    relMax_sdata[tid] = dev_accMat[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(unsigned int s=1; s < loc_dim; s*=2){
      if(tid % (2*s) == 0){ //it is for a different stride
	max_sdata[tid] = (max_sdata[tid] > max_sdata[tid+s]) ? max_sdata[tid] : max_sdata[tid+s];
	barrier(CLK_LOCAL_MEM_FENCE);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      
    }
    
    if(tid == 0) dev_output[get_group_id(0)] = max_sdata[0]; //at sdata[0], we found the maximum
    
    if(relMax_sdata[tid] >= ac_soglia){ 
      dev_maxRelOutput[i] = relMax_sdata[tid];
    }else{
      dev_maxRelOutput[i] = 0;
    }

  }

}

#endif