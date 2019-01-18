#!/bin/bash


dataset=0

for seed in $(seq 1 50); do
	for points in 530 550; do
		for tasks in 50 200 500; do
			for dims in 30; do
				for method in 5; do


					RAM=2.8G
					TIME=2:00:00
        			qsub -N ss.$seed.$method.$points.$tasks -l tmem=$RAM -l h_vmem=$RAM -l h_rt=$TIME python.sh $seed $points $tasks $dims $method $dataset

				done;
			done;
		done;
	done;
done;
