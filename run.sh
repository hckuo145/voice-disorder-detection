mode=naive
model=StdCNN
kfold_idx=15

device=cuda:$((kfold_idx%8))
echo ${device}

for ((test_fold=0; test_fold<5; test_fold++))
do
     title=${model}_${mode}_k${kfold_idx}f${test_fold}
     echo ${title}

     cmd="--title ${title} --device ${device} --batch 30    \
          --kfold_idx ${kfold_idx} --test_fold ${test_fold} \
          --model_conf config/model/${model}.yaml           \
          --hyper_conf config/hyper/${mode}.yaml"

     python main.py --train ${cmd}
done