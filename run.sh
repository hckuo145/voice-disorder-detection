kfold_idx=19
test_fold=4

device=cuda:$(((4 * ${kfold_idx} + ${test_fold} - 1) % 8))
echo ${device}


# title=StdCNN_naive_k${kfold_idx}f${test_fold}

# for ((seed=0; seed<200; seed++))
# do
#      cmd="--title ${title} --seed ${seed} --device ${device} --batch 30 \
#           --kfold_idx ${kfold_idx} --test_fold ${test_fold}             \
#           --model_conf config/model/StdCNN.yaml                         \
#           --hyper_conf config/hyper/naive.yaml                          \
#           --params exp/${title}/seed${seed}/ckpt/best_valid_src_uar.pt"

#      mkdir -p exp/${title}/seed${seed}
#      # logfile=exp/${title}/seed${seed}/history.log
     
#      python main.py --train ${cmd}
# done


title=SepCNN_naive_k${kfold_idx}f${test_fold}

for ((seed=0; seed<200; seed++))
do
     cmd="--title ${title} --seed ${seed} --device ${device} --batch 30 \
          --kfold_idx ${kfold_idx} --test_fold ${test_fold}             \
          --model_conf config/model/SepCNN.yaml                         \
          --hyper_conf config/hyper/naive.yaml                          \
          --params exp/${title}/seed${seed}/ckpt/best_valid_src_uar.pt"

     mkdir -p exp/${title}/seed${seed}
     # logfile=exp/${title}/seed${seed}/history.log

     python main.py --train ${cmd}
done