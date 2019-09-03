#!/bin/bash

for i in 100 200 300 400 500 1000 1500 2000 2500 3000 3500 4000 4500 5000
do
   nvprof --log-file %1 ./ht_rhophi -l 100 -n $i > PISA_output/with_UM/rhophi/output.rhophi-$i-K20m.txt
   nvprof --log-file %1 ./ht_AB -l 100 -n $i > PISA_output/with_UM/AB/output.AB-$i-K20m.txt
   nvprof --log-file %1 ./ht_rhophi-0.1 -l 100 -n $i > PISA_output/with_UM/rhophimgpu/output.rhophimgpu-$i-K20m.txt
   echo "done loop $i"
done
