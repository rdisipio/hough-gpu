#!/bin/bash

for i in 100 200 300 400 500 1000 1500 2000 2500 3000 3500 4000 4500 5000
do
   echo "$i hits"
   cat rhophi/output.rhophi-$i-K40m-100l.txt | grep voteHoughSpace
   cat rhophi/output.rhophi-$i-K40m-100l.txt | grep findRelativeMax_withShared

done
