
for ((i=0; i < 5; i++)); do 
	python dlib_reproduce_only_disentangled --model_num=$i
	
done
