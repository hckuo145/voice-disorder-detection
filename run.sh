# mode=adapt
# model=SepDACNN_a4
# kfold_idx=15

# device=cuda:$((kfold_idx%8))
# echo ${device}

# for ((test_fold=0; test_fold<5; test_fold++))
# do
#      title=${model}_${mode}_k${kfold_idx}f${test_fold}
#      echo -e "\n${title}"

#      cmd="--title ${title} --device ${device} --batch 30    \
#           --kfold_idx ${kfold_idx} --test_fold ${test_fold} \
#           --model_conf config/model/${model}.yaml           \
#           --hyper_conf config/hyper/${mode}.yaml"

#      python main.py --train ${cmd}
# done


# mode=finetune
# model=SepCNN
# kfold_idx=15

# device=cuda:$((kfold_idx%8))
# echo ${device}

# for ((test_fold=0; test_fold<5; test_fold++))
# do
#      title=${model}_${mode}_k${kfold_idx}f${test_fold}
#      params={exp_dir}/${model}_naive_k${kfold_idx}f${test_fold}/seed{seed}/ckpt/best_valid_src_uar.pt
#      echo -e "\n${title}"

#      cmd="--title ${title} --device ${device} --batch 30    \
#           --kfold_idx ${kfold_idx} --test_fold ${test_fold} \
#           --model_conf config/model/${model}.yaml           \
#           --hyper_conf config/hyper/${mode}.yaml            \
#           --params ${params}"

#      python main.py --train --finetune ${cmd}
# done


mode=adapt
model=SepDACNN_a2

for ((kfold_idx=0; kfold_idx<16; kfold_idx++))
do
     for ((test_fold=0; test_fold<5; test_fold++))
     do
          title=${model}_${mode}_k${kfold_idx}f${test_fold}
          params={exp_dir}/${title}/seed{seed}/ckpt/best_valid_avg_uar.pt
          echo -e "\n${title}"

          cmd="--title ${title} --device cuda:0 --batch 30       \
               --kfold_idx ${kfold_idx} --test_fold ${test_fold} \
               --model_conf config/model/${model}.yaml           \
               --hyper_conf config/hyper/${mode}.yaml            \
               --logfile {exp_dir}/${title}/record.txt           \
               --params ${params}"

          python main.py --test ${cmd}

          # rm exp/${title}/record.txt
     done
done



# mode=naive
# model=StdCNN
# kfold_idx=0
# test_fold=0

# device=cuda:0
# echo ${device}

# title=${model}_${mode}_k${kfold_idx}f${test_fold}
# echo -e "\n${title}"

# cmd="--title ${title} --device ${device}               \
#      --kfold_idx ${kfold_idx} --test_fold ${test_fold} \
#      --model_conf config/model/${model}.yaml           \
#      --hyper_conf config/hyper/${mode}.yaml"

# python main.py --eval ${cmd}