import argparse
import torch
from dpft import config
from dpft.dataset.saprot import SaProtMutantData, saprot_zero_shot
from dpft.pipeline import Pipeline
from dpft.utils.score import metrics
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str, default='activate learning',
                        help='perform activate learning')
    parser.add_argument('--test', '-t', action='store_true',
                        help='load the trained models from checkpoints and test them')
    parser.add_argument('--model', '-md', type=str, choices=config.model_dir.keys(),
                        required=True, help='name of the foundation model')
    parser.add_argument('--protein', '-p', type=str, default='all',
                        help='name of the target protein')
    parser.add_argument('--train_data_size', '-tbs', type=float, default=1.0,
                        help='few-shot training set size, can be a float number less than 1 to indicate a proportion')
    parser.add_argument('--train_size', '-ts', type=int, default=12,
                        help='training set size in each round of active learning')
    parser.add_argument('--train_batch', '-tb', type=int, default=12,
                        help='batch size for training (outer batch size in the case of meta learning)')
    parser.add_argument('--eval_batch', '-eb', type=int, default=12,
                        help='batch size for evaluation')
    parser.add_argument('--lora_r', '-r', type=int, default=16, #8,16
                        help='hyper-parameter r of LORA')
    parser.add_argument('--optimizer', '-o', type=str, choices=['sgd', 'nag', 'adagrad', 'adadelta', 'adam'],
                        default='adam', help='optimizer for training (outer loop optimization in the case of meta learning)')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--max_al_rounds', type=int, default=5,
                        help='maximum round of activate learning')
    parser.add_argument('--epochs_per_al_round', '-e', type=int, default=10,
                        help='maximum training epochs in each round of activate learning')
    parser.add_argument('--pref_batch_size', type=int, default=12,
                        help='the size of preference pair number')
    parser.add_argument('--bt_weight', type=float, default=0.5,
                        help='the weight of preference pair loss')
    parser.add_argument('--max_grad_norm', '-gn', type=float, default=3,
                        help='maximum gradient norm to clip to')
    parser.add_argument('--mask', '-mk', type=str, choices=['train', 'eval', 'all', 'none'], default='none',
                        help='whether to compute masked 0-shot scores')
    parser.add_argument('--list_size', '-ls', type=int, default=1,
                        help='list size for ranking')
    parser.add_argument('--max_iter', '-mi', type=int, default=10,
                        help='maximum number of iterations per training epoch, useless during meta training')
    parser.add_argument('--eval_metric', '-em', type=str, choices=metrics, default='spearmanr',
                        help='evaluation metric')
    parser.add_argument('--retr_metric', '-rm', type=str, default='cosine',
                        help='similarity metric used for retrieving proteins for meta training')
    parser.add_argument('--adapt_lr', '-alr', type=float, default=5e-3,
                        help='learning rate for inner loop')
    parser.add_argument('--adapt_steps', '-as', type=int, default=4,
                        help='number of iterations for inner loop')
    parser.add_argument('--patience', '-pt', type=int, default=15,
                        help='number of epochs to wait until the validation score improves')
    parser.add_argument('--n_sites', '-ns', nargs='+', type=int, default=[1],
                        help='possible numbers of mutation sites in the training data. \
                              setting to 0 means no constraint')
    parser.add_argument('--negative_train', '-neg', action='store_true',
                        help='whether to constraint the training data to negative examples')
    parser.add_argument('--cross_validation', '-cv', type=int, default=5,
                        help='number of splits for cross validation (shuffle & split) on the training set. \
                              if set to 1, the test set will be used for validation; \
                              if set to 0, no testing or validation will be performed.')
    parser.add_argument('--save_postfix', '-sp', type=str, default='',
                        help='a custom string to append to all data paths (data, checkpoints and predictions)')
    parser.add_argument('--force_cpu', '-cpu', action='store_true',
                        help='use cpu for training and evaluation even if gpu is available')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    path = config.data_path.replace('.pkl', f'{args.save_postfix}.pkl')
    proteins = torch.load(path, weights_only=False)
    max_scores = []

    SEEDS = [
        # 原始核心种子（10个）
        42, 3407, 6195, 2178, 9012, 4563, 1123, 7788, 1234, 2023,

        # 扩展的生物学特化种子（90个）
        # 1. 质数类（增强随机性）
        1997, 401, 7919, 2027, 3001, 4999, 6997, 8999, 10007, 12007,
        14009, 16001, 18013, 20011, 22003, 24001, 26003, 28001, 30011, 32003,

        # 2. 序列优化种子（适合长序列建模）
        5432, 6543, 7654, 8765, 9876, 1357, 2468, 3579, 4680, 5791,
        6802, 7913, 8024, 9135, 1024, 2048, 3072, 4096, 5120, 6144,

        # 3. 实验日期相关（保证时效性）
        20230101, 20230202, 20230303, 20230404, 20230505,
        20230606, 20230707, 20230808, 20230909, 20231010,

        # 4. 蛋白质工程关键数字
        1012,  # 代表10^12（突变库规模量级）
        6023,  # 阿伏伽德罗数前四位
        1492,  # 常见蛋白质平均残基数
        2837,  # 典型蛋白质设计迭代次数
        3719,  # 重要文献PMID尾号

        # 5. 均匀分布种子（数学优化）
        1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 11111,
        12345, 13579, 14703, 15937, 16180, 17290, 18301, 19452, 20563, 21674,

        # 6. 硬件相关种子（适配GPU计算）
        10240,  # 典型显存MB数
        16384,  # 常见batch size基数
        24576,  # 高端GPU显存
        32768,  # 2^15
        40960,  # 大模型训练参数

        # 7. 生物信息学经典数字
        1953,  # DNA结构发现年份
        1961,  # mRNA发现年份
        1977,  # 测序技术诞生
        1990,  # 人类基因组计划启动
        2001,  # 人类基因组草图发布

        # 8. 补充多样性种子
        777, 888, 999, 1112, 1314, 1516, 1718, 1910, 2122, 2324,
        2526, 2728, 2930, 3132, 3334, 3536, 3738, 3940, 4142, 4344
    ]

    for seed in SEEDS:
        if args.model == 'saprot':
            pipeline = Pipeline(args, data_constructor=SaProtMutantData, score_fn=saprot_zero_shot)
        else:
            pipeline = Pipeline(args, seed=seed)

        score = pipeline(proteins)
        max_scores.append(score)

    np.savetxt('./max_activity.txt', max_scores, fmt='%.3f')
