kfold_idx=20

for ((test_fold=0; test_fold<5; test_fold++))
do
     title=CNN_naive_k${kfold_idx}f${test_fold}

     for ((seed=0; seed<200; seed++))
     do
          cmd="--title $title --seed $seed --device cuda --batch 30 \
               --kfold_idx $kfold_idx --test_fold $test_fold        \
               --model_conf config/model/CNN.yaml                   \
               --hyper_conf config/hyper/naive.yaml                 \
               --params exp/$title/seed$seed/ckpt/best_valid_src_uar.pt"

          mkdir -p exp/${title}/seed${seed}
          logfile=exp/${title}/seed${seed}/history.log

          python main.py --train $cmd > $logfile
          # python main.py --test  $cmd
     done
done