kfold_idx=19
device=cuda:$((${kfold_idx}%4))

# for ((test_fold=0; test_fold<5; test_fold++))
# do
#      title=StdCNN_naive_k${kfold_idx}f${test_fold}

#      for ((seed=0; seed<200; seed++))
#      do
#           cmd="--title ${title} --seed ${seed} --device ${device} --batch 30 \
#                --kfold_idx ${kfold_idx} --test_fold ${test_fold}             \
#                --model_conf config/model/StdCNN.yaml                         \
#                --hyper_conf config/hyper/naive.yaml                          \
#                --params exp/${title}/seed${seed}/ckpt/best_valid_src_uar.pt"

#           mkdir -p exp/${title}/seed${seed}
#           logfile=exp/${title}/seed${seed}/history.log

#           python main.py --train ${cmd} > ${logfile}
#      done
# done

for ((test_fold=0; test_fold<5; test_fold++))
do
     title=SepCNN_naive_k${kfold_idx}f${test_fold}

     for ((seed=0; seed<200; seed++))
     do
          cmd="--title ${title} --seed ${seed} --device ${device} --batch 30 \
               --kfold_idx ${kfold_idx} --test_fold ${test_fold}             \
               --model_conf config/model/SepCNN.yaml                         \
               --hyper_conf config/hyper/naive.yaml                          \
               --params exp/${title}/seed${seed}/ckpt/best_valid_src_uar.pt"

          mkdir -p exp/${title}/seed${seed}
          logfile=exp/${title}/seed${seed}/history.log

          python main.py --train ${cmd} > ${logfile}
     done
done