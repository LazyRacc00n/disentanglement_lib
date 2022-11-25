for ((i=0; i < 5; i++)); do 
	python dlib_evaluate_only_disentangled.py --model_num=$i
	
done
