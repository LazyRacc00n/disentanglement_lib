

for ((i=1; i < 12; i++)); do 
	python dlib_evaluate_my_initial_experiment --model_num=$i
	
done
