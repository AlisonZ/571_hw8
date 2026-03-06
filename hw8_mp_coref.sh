#!/bin/bash
# 1: <training_file> ./data/train_pairs.jsonx
#2: <embedding_file> ./dolma_300_2024_1.2M.100_combined.txt
# 3: <testing_file> ./data/test_pairs.json
# 4: <output_vector_file>: hw8_vectors_output.txt
#5: <output_classification_filename>: hw8_classification_output.txt
# ./hw8_mp_coref.sh . /data/toy_test_pairs.json hw8_vectors_output.txt hw8_classification_output.txt
# ./hw8_mp_coref.sh /mnt/dropbox/25-26/571W/hw8/data/train_pairs.json /mnt/dropbox/25-26/571W/hw8/dolma_300_2024_1.2M.100_combined.txt /mnt/dropbox/25-26/571W/hw8/data/test_pairs.json hw8_vectors_output.txt hw8_classification_output.txt
python3 hw8_mp_coref.py $1 $2 $3 $4 $5