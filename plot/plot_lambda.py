import matplotlib.pyplot as plt

test_fold = 3

infiles = [ 
    f'../exp/SepDACNN_a001_adapt_k11f{test_fold}/record.txt',
    f'../exp/SepDACNN_a005_adapt_k11f{test_fold}/record.txt',
    f'../exp/SepDACNN_a01_adapt_k11f{test_fold}/record.txt' ,
    f'../exp/SepDACNN_a05_adapt_k11f{test_fold}/record.txt' ,
    f'../exp/SepDACNN_a1_adapt_k11f{test_fold}/record.txt'  ,
    f'../exp/SepDACNN_a5_adapt_k11f{test_fold}/record.txt'  ,
    f'../exp/SepDACNN_a10_adapt_k11f{test_fold}/record.txt'   
]

src_scores = []
tgt_scores = []
for infile in infiles:
    with open(infile, 'r') as f:
            lines = f.readlines()

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

    src_scores.append([ seed_dict[seed]['test_src_uar'] for seed in seed_dict ])
    tgt_scores.append([ seed_dict[seed]['test_tgt_uar'] for seed in seed_dict ])


c1 = 'lightcoral'
c2 = 'lightseagreen'
fig, ax = plt.subplots()
src_bp = ax.boxplot(src_scores, positions=[1,4,7,10,13,16,19],  showfliers=False, notch=True, patch_artist=True, \
    boxprops=dict(facecolor=c1), medianprops=dict(color=c1))
tgt_bp = ax.boxplot(tgt_scores, positions=[2,5,8,11,14,17,20], showfliers=False, notch=True, patch_artist=True, \
    boxprops=dict(facecolor=c2), medianprops=dict(color=c2))
ax.legend([src_bp['boxes'][0], tgt_bp['boxes'][0]], ['source\n(clean)', 'target\n(noisy)'], loc='upper right')

plt.xticks(ticks=[1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5], labels=[0.01, 0.05, 0.1, 0.5, 1, 5, 10])
plt.xlabel(r'$\lambda$')
plt.ylabel('UAR')
plt.title(r'Ablation: The Effect of $\lambda$')

plt.tight_layout()
plt.savefig(f'lambda_k11f{test_fold}.png', dpi=600)
plt.close()


infiles = [ 
    f'../exp/SepDACNN_a1_adapt_k11f{test_fold}/record.txt',
    f'../exp/SepDACNN_a2_adapt_k11f{test_fold}/record.txt',
    f'../exp/SepDACNN_a3_adapt_k11f{test_fold}/record.txt',
    f'../exp/SepDACNN_a4_adapt_k11f{test_fold}/record.txt',
    f'../exp/SepDACNN_a5_adapt_k11f{test_fold}/record.txt'
]

src_scores = []
tgt_scores = []
for infile in infiles:
    with open(infile, 'r') as f:
            lines = f.readlines()

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

    src_scores.append([ seed_dict[seed]['test_src_uar'] for seed in seed_dict ])
    tgt_scores.append([ seed_dict[seed]['test_tgt_uar'] for seed in seed_dict ])


c1 = 'lightcoral'
c2 = 'lightseagreen'
fig, ax = plt.subplots()
src_bp = ax.boxplot(src_scores, positions=[1,4,7,10,13],  showfliers=False, notch=True, patch_artist=True, \
    boxprops=dict(facecolor=c1), medianprops=dict(color=c1))
tgt_bp = ax.boxplot(tgt_scores, positions=[2,5,8,11,14], showfliers=False, notch=True, patch_artist=True, \
    boxprops=dict(facecolor=c2), medianprops=dict(color=c2))
ax.legend([src_bp['boxes'][0], tgt_bp['boxes'][0]], ['source\n(clean)', 'target\n(noisy)'], loc='upper right')

plt.xticks(ticks=[1.5, 4.5, 7.5, 10.5, 13.5], labels=[1, 2, 3 ,4, 5])
plt.xlabel(r'$\lambda$')
plt.ylabel('UAR')
plt.title(r'Ablation: The Effect of $\lambda$ form 1 to 5')

plt.tight_layout()
plt.savefig(f'lambda_k11f{test_fold}_1to5.png', dpi=600)
plt.close()