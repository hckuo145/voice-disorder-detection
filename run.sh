kfold_idx=0
test_fold=0

# device=cuda:$(((4 * ${kfold_idx} + ${test_fold} - 1) % 8))
device=cuda
echo ${device}


title=StdCNN_naive_k${kfold_idx}f${test_fold}

cmd="--title ${title} --device ${device} --batch 30    \
     --kfold_idx ${kfold_idx} --test_fold ${test_fold} \
     --model_conf config/model/StdCNN.yaml             \
     --hyper_conf config/hyper/naive.yaml"
     
python main.py --train ${cmd}



# title=SepCNN_naive_k${kfold_idx}f${test_fold}

# cmd="--title ${title} --device ${device} --batch 30    \
#      --kfold_idx ${kfold_idx} --test_fold ${test_fold} \
#      --model_conf config/model/SepCNN.yaml             \
#      --hyper_conf config/hyper/naive.yaml"                         

# python main.py --train ${cmd}