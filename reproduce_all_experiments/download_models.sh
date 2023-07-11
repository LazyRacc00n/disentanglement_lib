for ((i=0; i < 300; i++)); do 
	wget https://storage.googleapis.com/disentanglement_lib/unsupervised_study_v1/$i.zip
        unzip $i.zip -d "output/unsupervised_study_v1"
        rm -r $i.zip
	
done
