from collections import defaultdict

# # title   = 'exp/SepCNN_target'
# # title   = 'exp/SepCNN_naive'
# # monitor = 'test_src_uar'

# # title   = 'exp/SepCNN_joint'
# title   = 'exp/SepDACNN_a05_adapt'
# monitor = 'test_avg_uar'

# # title   = 'exp/SepCNN_finetune'
# # monitor = 'test_tgt_uar'

# fold_dict = {}
# # for kfold_idx in range(16):
# for kfold_idx in [11, 10, 13, 15, 12, 3, 5, 6, 4, 8]:
#     fold_dict[f'{title}_k{kfold_idx}'] = defaultdict(float)
    
#     for test_fold in range(5):
#         with open(f'{title}_k{kfold_idx}f{test_fold}/record.txt', 'r') as infile:
#             lines = infile.readlines()

#         seed_dict = {}
#         for line in lines:
#             info = line.strip().split(' - ')
#             for i, entry in enumerate(info):
#                 if i == 0:
#                     seed = entry
#                     seed_dict[seed] = {}
#                 else:
#                     key, val = entry.split(': ')
#                     seed_dict[seed][key] = float(val)

#         sorted_seeds = sorted(seed_dict.keys(), key=lambda name: seed_dict[name][monitor], reverse=True)
#         for key, val in seed_dict[sorted_seeds[0]].items():
#             fold_dict[f'{title}_k{kfold_idx}'][key] += val / 5

# sorted_folds = sorted(fold_dict.keys(), key=lambda name: fold_dict[name][monitor], reverse=True)
# for fold in sorted_folds:
#     print(f'{fold:<27}', end=' ')
#     for key, val in fold_dict[fold].items():
#         print(f'{key}: {val:5.2f},', end=' ')
#     print('\b\b ')

# print('')
# score_dict = defaultdict(float)
# for i, fold in enumerate(sorted_folds):
#     for key, val in fold_dict[fold].items():
#         score_dict[key] += val

# #     if i == 2:
# #         print(f'{"Top 3":<27}', end=' ')
# #         for key, val in score_dict.items():
# #             print(f'{key}: {val / 3:5.2f},', end=' ')
# #         print('\b\b ')

# #     if i == 4:
# #         print(f'{"Top 5":<27}', end=' ')
# #         for key, val in score_dict.items():
# #             print(f'{key}: {val / 5:5.2f},', end=' ')
# #         print('\b\b ')
    
# #     if i == 9:    
# #         print(f'{"Top 10":<27}', end=' ')
# #         for key, val in score_dict.items():
# #             print(f'{key}: {val / 10:5.2f},', end=' ')
# #         print('\b\b ')

# # print(f'{"All 16":<27}', end=' ')
# # for key, val in score_dict.items():
# #     print(f'{key}: {val / 16:5.2f},', end=' ')
# # print('\b\b ')

# print(f'{"All 10":<27}', end=' ')
# for key, val in score_dict.items():
#     print(f'{key}: {val / 10:5.2f},', end=' ')
# print('\b\b ')


from scipy import stats

# title   = 'exp/StdCNN_naive'
# monitor = 'test_src_uar'
title   = 'exp/SepDACNN_a05_adapt'
monitor = 'test_avg_uar'

UAR1 = {'src': [], 'tgt': []}
for kfold_idx in [11, 10, 13, 15, 12, 3, 5, 6, 4, 8]:
    
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
        
        UAR1['src'].append(seed_dict[sorted_seeds[0]]['test_src_uar'])
        UAR1['tgt'].append(seed_dict[sorted_seeds[0]]['test_tgt_uar'])


title   = 'exp/StdCNN_naive'
monitor = 'test_src_uar'
# title   = 'exp/SepCNN_naive'
# monitor = 'test_src_uar'
# title   = 'exp/SepCNN_target'
# monitor = 'test_src_uar'
# title   = 'exp/SepCNN_finetune'
# monitor = 'test_tgt_uar'
# title   = 'exp/SepCNN_joint'
# monitor = 'test_avg_uar'

UAR2 = {'src': [], 'tgt': []}
for kfold_idx in [11, 10, 13, 15, 12, 3, 5, 6, 4, 8]:
    
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
        
        UAR2['src'].append(seed_dict[sorted_seeds[0]]['test_src_uar'])
        UAR2['tgt'].append(seed_dict[sorted_seeds[0]]['test_tgt_uar'])

_, src_p = stats.ttest_ind(UAR1['src'], UAR2['src'])
_, tgt_p = stats.ttest_ind(UAR1['tgt'], UAR2['tgt'])
print(src_p, tgt_p)