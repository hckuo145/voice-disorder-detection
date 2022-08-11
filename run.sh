title=CNN_naive_k9f0

cmd="--title $title --device cuda --batch 32 \
     --kfold_idx 9 --test_fold 0 --seed 0    \
     --model_conf config/model/CNN.yaml      \
     --hyper_conf config/hyper/naive.yaml    \
     --params exp/$title/ckpt/best_valid_source_uar.pt"

mkdir -p exp/$title/seed0
logfile=exp/$title/seed0/history.log

python main.py --train $cmd > $logfile
# python main.py --test $cmd