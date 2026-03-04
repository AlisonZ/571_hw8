#!/bin/bash
# 1: <training_file> ./data/train_pairs.json
# 2: <testing_file> ./data/test_pairs.json
# 3: <output_vector_file>: hw8_vectors_output.txt
#4: <output_classification_filename>: hw8_classification_output.txt
# ./hw8_mp_coref.sh ./data/train_pairs.json /data/test_pairs.json hw8_vectors_output.txt hw8_classification_output.txt
python3 hw8_mp_coref.py $1 $2 $3 $4