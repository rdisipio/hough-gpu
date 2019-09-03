#!/bin/bash

for i in 100 200 300 400 500 1000 1500 2000 2500 3000 3500 4000 4500 5000
do
   nvprof --log-file ATLAS_output/rhophi/output.rhophi-$i-K20m-100l.txt ./ht_rhophi -l 100 -n $i 
   nvprof --log-file ATLAS_output/AB/output.AB-$i-K20m-100l.txt ./ht_AB -l 100 -n $i 
   nvprof --log-file ATLAS_output/rhophimgpu/output.rhophimgpu-$i-K20m-100l.txt ./ht_rhophi-0.1 -l 100 -n $i
   echo "done loop $i"
done
