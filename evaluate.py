from collections import defaultdict

title   = 'exp/SepDACNN_a5_adapt'
monitor = 'test_avg_uar'

fold_dict = {}
for kfold_idx in range(16):
    fold_dict[f'{title}_k{kfold_idx}'] = defaultdict(float)
    
    for test_fold in range(5):
        with open(f'{title}_k{kfold_idx}f{test_fold}/record.txt', 'r') as infile:
            lines = infile.readlines()

        seed_dict = {}
        for line in lines:
            info = line.strip().split(' - ')
            for i, entry in enumerate(info):
                if i == 0:
                    seed = entry
                    seed_dict[seed] = {}
                else:
                    key, val = entry.split(': ')
                    seed_dict[seed][key] = float(val)

        sorted_seeds = sorted(seed_dict.keys(), key=lambda name: seed_dict[name][monitor], reverse=True)

        for key, val in seed_dict[sorted_seeds[0]].items():
            fold_dict[f'{title}_k{kfold_idx}'][key] += val / 5

# print(fold_dict)
sorted_folds = sorted(fold_dict.keys(), key=lambda name: fold_dict[name][monitor], reverse=True)
for fold in sorted_folds:
    print(f'{fold:<27}', end=' ')
    for key, val in fold_dict[fold].items():
        print(f'{key}: {val:5.2f},', end=' ')
    print('')

print('')
score_dict = defaultdict(float)
for i, fold in enumerate(sorted_folds):
    for key, val in fold_dict[fold].items():
        score_dict[key] += val

    if i == 2:
        print(f'{"Top3":<27}', end=' ')
        for key, val in score_dict.items():
            print(f'{key}: {val / 3:5.2f},', end=' ')
        print('')

    if i == 4:
        print(f'{"Top5":<27}', end=' ')
        for key, val in score_dict.items():
            print(f'{key}: {val / 5:5.2f},', end=' ')
        print('')
    
    if i == 9:    
        print(f'{"Top10":<27}', end=' ')
        for key, val in score_dict.items():
            print(f'{key}: {val / 10:5.2f},', end=' ')
        print('')

print(f'{"All16":<27}', end=' ')
for key, val in score_dict.items():
    print(f'{key}: {val / 16:5.2f},', end=' ')
print('')