//
//  ht_helix.cpp
//  
//
//  Created by Lorenzo Rinaldi on 29/04/14.
//
//


#include <string.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include <cuda_runtime.h>
// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper utility functions 

using namespace std;

#define NHMAX 300
#define Nsec 4 // Numero settori in piano trasverso
#define Ntheta 16 // Numero settori in piano longitudinale
#define Nphi 2000 // Numero bin angolo polare
#define Nrho 2000 // Numero bin distanza radiale

#define rhomin 500.f // mm
#define rhomax 100000.f // mm
#define phimin 0.f // rad
#define phimax 2*M_PI // rad
#define thetamin 0.f // rad
#define thetamax M_PI // rad

#define ac_soglia 4 // soglia nella matrice di accumulazione


int acc_Mat [ Nsec ][ Ntheta ][Nphi ] [Nrho ];
//int Max_rel [ Nsec ][ Ntheta ][Nphi ] [Nrho ];


float dtheta=M_PI/Ntheta;
float drho=(rhomax-rhomin)/Nrho;
float dphi=(phimax-phimin)/Nphi;

vector<float> x_values;
vector<float> y_values;
vector<float> z_values;

void read_inputFile(string file_path);

__global__ void voteHoughSpace(float *dev_x_values, float *dev_y_values, float *dev_z_values, int *dev_accMat, float dtheta, float drho, float dphi){
  
  unsigned int index = blockIdx.x;
  
  float R2 = dev_x_values[index]*dev_x_values[index] + dev_y_values[index]*dev_y_values[index];
  float theta=acos(dev_z_values[index]/(R2+dev_z_values[index]*dev_z_values[index]));
  
  int ith=(int) (theta/dtheta)+0.5f;
  
  float sec=atan2(dev_y_values[index],dev_x_values[index]);
  if (sec<0.f)
  {
    sec=2*M_PI+sec;
  }
  int isec=int(sec/2/M_PI*Nsec);
  
  int iphi = threadIdx.x;
  float phi=phimin+iphi*dphi;
  float rho=R2/2.f/(dev_x_values[index]*cos(phi)+dev_y_values[index]*sin(phi));
  int irho=(int)((rho-rhomin)/drho)+0.5f;
  if (rho<=rhomax && rho>rhomin)
  {
    acc_Mat[isec][ith][iphi][irho]++;
  }
}

int main(int argc, char* argv[]){
    
  
    int *dev_accMat;
    float *dev_x_values;
    float *dev_y_values;
    float *dev_z_values;
    
    float *x_values_temp;
    float *y_values_temp;
    float *z_values_temp;
    //float R = 0.f;
    
    int debug_accMat[ Nsec ][ Ntheta ][Nphi ] [Nrho ];
    
    // Inizializzo a zero le matrici
    //memset(&acc_Mat, 0, (sizeof(int)*(Nsec*Ntheta*Nphi*Nrho)) );
    //memset(&Max_rel, 0, (sizeof(int)*(Nsec*Ntheta*Nphi*Nrho)) );
    
    //alloc accumulator matrix on GPU
    cudaMalloc((void **) &dev_accMat, (sizeof(int)*(Nsec*Ntheta*Nphi*Nrho)) );
    cudaMemset(dev_accMat, 0, (sizeof(int)*(Nsec*Ntheta*Nphi*Nrho)));
    
   
    
    //riempi i valori dentro x_values , y_values , z_values
    read_inputFile("hits1.txt");
//    read_inputFile("../datafiles/hits-1.txt");
    
    cudaMalloc((void **) &dev_x_values, sizeof(float)*x_values.size());
    cudaMalloc((void **) &dev_y_values, sizeof(float)*y_values.size());
    cudaMalloc((void **) &dev_z_values, sizeof(float)*z_values.size());
    
    x_values_temp = (float*) malloc(sizeof(float)*x_values.size());
    y_values_temp =  (float*) malloc(sizeof(float)*y_values.size());
    z_values_temp = (float*)  malloc( sizeof(float)*z_values.size());
    
    for(unsigned int i = 0; i < x_values.size(); i++){
      x_values_temp[i] = x_values.at(i);
      y_values_temp[i] = y_values.at(i);
      z_values_temp[i] = z_values.at(i);
    }
    
    cudaMemcpy(dev_x_values, x_values_temp, sizeof(float)*x_values.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y_values, y_values_temp, sizeof(float)*y_values.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_z_values, z_values_temp, sizeof(float)*z_values.size(), cudaMemcpyHostToDevice);
    
    voteHoughSpace <<<x_values.size(), Nphi>>> (dev_x_values, dev_y_values, dev_z_values, dev_accMat, dtheta, drho, dphi);
    
    cudaMemcpy(&debug_accMat, dev_accMat, (sizeof(int)*(Nsec*Ntheta*Nphi*Nrho)), cudaMemcpyDeviceToHost);
    
    for(unsigned int i = 0; i < x_values.size(); i++){
        //cout << x_values.at(i) << " - ";
        //cout << y_values.at(i) << endl;
        
        float R2=x_values.at(i)*x_values.at(i)+y_values.at(i)*y_values.at(i);
        float theta=acos(z_values.at(i)/(R2+z_values.at(i)*z_values.at(i)));
        int ith=(int) (theta/dtheta)+0.5f;
        
        float sec=atan2(y_values.at(i),x_values.at(i));
        if (sec<0.f)
        {
            sec=2*M_PI+sec;
        }
        int isec=int(sec/2/M_PI*Nsec);
        
        for(int iphi = 0; iphi < Nphi; iphi++){
            float phi=phimin+iphi*dphi;
            float rho=R2/2.f/(x_values.at(i)*cos(phi)+y_values.at(i)*sin(phi));
            int irho=(int)((rho-rhomin)/drho)+0.5f;
            if (rho<=rhomax && rho>rhomin)
            {
                acc_Mat[isec][ith][iphi][irho]++;
            }
        }
    }
    //check
    for(unsigned int isec = 0; isec < Nsec; isec++){
        
        for(unsigned int ith = 1; ith < Ntheta; ith++){
            
            for(unsigned int iphi = 1; iphi < Nphi; iphi++){
                
                for(unsigned int irho = 1; irho < Nrho; irho++){
		  
		  acc_Mat[isec][ith][iphi][irho] != debug_accMat[isec][ith][iphi][irho];
		  printf("diverso acc_Mat[%d][%d][%d][%d] %d - debug_accMat[%d][%d][%d][%d] %d \n", isec, ith, iphi, irho, acc_Mat[isec][ith][iphi][irho],
		    isec, ith, iphi, irho, debug_accMat[isec][ith][iphi][irho]);
                }
	    }
	}
    }
    
    //trova il massimo relativo
    
    int accumax = -1;
    int iphiMax = 0;
    int irhoMax = 0;
    int ithMax = 0;
    int isecMax = 0;
    int NMrel =0 ;
    
    for(unsigned int isec = 0; isec < Nsec; isec++){
        
        for(unsigned int ith = 1; ith < Ntheta; ith++){
            
            for(unsigned int iphi = 1; iphi < Nphi; iphi++){
                
                for(unsigned int irho = 1; irho < Nrho; irho++){
                    
                    float acc=acc_Mat[isec][ith][iphi][irho];
                    if (acc >= ac_soglia){
                        if (acc > accumax){
                            accumax=acc;
                        }
                        if (acc>acc_Mat[isec][ith-1][iphi][irho] && acc >= acc_Mat[isec][ith+1][iphi][irho]){
                            if (acc>acc_Mat[isec][ith][iphi-1][irho-1] && acc >= acc_Mat[isec][ith][iphi+1][irho+1]){
                                if (acc>acc_Mat[isec][ith][iphi][irho-1] && acc >= acc_Mat[isec][ith][iphi][irho+1]){
                                    if (acc>acc_Mat[isec][ith][iphi+1][irho-1] && acc >= acc_Mat[isec][ith][iphi+1][irho+1]){
                                        if (acc>=acc_Mat[isec][ith][iphi+1][irho] ){
                                            accumax = acc_Mat[isec][ith][iphi+1][irho];
                                            //Max_rel[isec][ith][iphi+1][irho]=1;
                                            NMrel++;
                                            ithMax=ith;
                                            irhoMax=irho;
                                            iphiMax=iphi;
                                            isecMax=isec+1;
                                            float t_th=(thetamin+ithMax*dtheta)*360.f/M_PI;
                                            float t_rho=rhomin+irhoMax*drho;
                                            float t_phi=phimin+iphiMax*dphi;
                                            //float q=t_rho/sin(t_phi);
                                            //float xm=-1/tan(t_phi);
                                            cout << acc <<" "<< t_rho <<" "<< t_phi << " " << isecMax << endl;
                                            
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    
    return 0;
}

/*****************************
 * file opener
 *****************************/


void read_inputFile(string file_path)
{
    
    ifstream input_f;
    
    string line;
    string value;
    
    stringstream ss;
    unsigned int val_iter;
    
    input_f.open(file_path.c_str());
    
    if (input_f.is_open())
    {
        while ( getline (input_f,line) )
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
        }
        input_f.close();
    }
    
    
    
    
}

