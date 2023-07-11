for ((i=0; i < 300; i++)); do 
        
        path="output/unsupervised_study_v1/${i}/postprocessed/mean"
	python3 one_diff_couple.py --path $path  --file_classes "train_factors_${i}"  --file_representation "train_representation_${i}"
        mv "${path}/X_train_representation_${i}.npz" "couples_representations/"
	mv "${path}/Y_train_representation_${i}.npz" "couples_representations/"
	
done
