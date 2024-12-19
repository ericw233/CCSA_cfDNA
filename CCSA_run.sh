#!/bin/bash
feature_type_list=(Frag Arm Griffin Cnv GMM Gemini MCMS Ma)
input_size=(1200 950 2600 2500 300 60 180 25)
output_path=/mnt/binf/eric/CCSA_May2024Results/CCSA_0516v2/
data_dir=/mnt/binf/eric/Mercury_Dec2023/Feature_all_Apr2024_frozenassource_v2.pkl

batch_size=256
epoch_num=2
batch_patience=300
alpha=0.5

for i in {0..4}
do
    python ./main_v1.py ${feature_type_list[i]} ${input_size[i]} ${data_dir} ${output_path} ${batch_size} ${epoch_num} ${batch_patience} ${alpha}
done
