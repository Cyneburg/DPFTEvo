import torch
import random
import numpy as np
import pandas as pd
from math import ceil
from itertools import chain
from transformers import AutoTokenizer, AutoModelForMaskedLM, EsmTokenizer, EsmForMaskedLM
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import DataLoader
from . import config
from .trainer import RankingTrainer 
from .dataset.base import MutantSequenceData, RankingSequenceData, generate_pref_pairs_from_scores
from .utils.data import make_dir, split_data
from .utils.score import metrics, group_scores, summarize_scores
import os
import datetime
import matplotlib.pyplot as plt

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def print_trainable_params(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f'Trainable params: {trainable_params} ({100 * trainable_params / all_param:.2f}%)')
    print(f'All params: {all_param}')

class Pipeline():
    def __init__(self, parsed_args, data_constructor=MutantSequenceData,
                 lora_modules=config.lora_modules, score_fn=None, seed=None):
        if parsed_args.n_sites == [0]:
            parsed_args.n_sites = None
        if not 0 < parsed_args.train_size < 1:
            parsed_args.train_size = int(parsed_args.train_size)
        self.args = parsed_args
        self.device = 'cuda:0' if  torch.cuda.is_available() else 'cpu'
        self.data_constructor = data_constructor
        self.lora_modules = lora_modules
        self.score_fn = score_fn
        self.get_cv_size = lambda train: 0.75 if len(train['df']) > 50 else 0.5
        self.report = dict()
        self.seed = seed
        set_seed(self.seed)
    
    def get_base_model(self, load_dir='/facebook/esm2_t36_3B_UR50D'): #esm2_t33_650M_UR50D
        print("model dir: {}".format(load_dir)) 
        args = self.args
        model_name = config.model_dir[args.model]
        if load_dir is None:
            model = EsmForMaskedLM.from_pretrained(model_name)
            for name, param in model.named_parameters():
                if 'contact_head.regression' in name:
                    param.requires_grad = False
        else:
            model = AutoModelForMaskedLM.from_pretrained(load_dir,torch_dtype='auto',device_map=None)
        tokenizer = AutoTokenizer.from_pretrained(load_dir)
        return model, tokenizer

    def get_save_dir(self, prefix, protein_name, prediction=False):
        args = self.args
        save_dir = '{}/{}/{}/{}/{}/{}/{}'.format(
            config.pred_dir if prediction else config.ckpt_dir,
            prefix,
            args.model,
            protein_name,
            datetime.datetime.now().strftime("%Y-%m-%d"),
            str(self.seed),
            args.save_postfix)
        return save_dir

    def _get_topk_samples(self, model, dataset, tokenizer, k=12):
        """预测未标注数据并返回Top-K高分的样本索引"""
        al_data = self.data_constructor(
            dataset, tokenizer,
            mask=self.args.mask in {'train', 'all'},
            device=self.device
        )

        al_iter = DataLoader(al_data, batch_size=self.args.eval_batch, collate_fn=al_data.collate)
        trainer = RankingTrainer(model, log_metrics=[], score_fn=self.score_fn)
        scores = []
        with torch.no_grad():
            for batch in al_iter:
                scores.append(trainer.predict(batch).cpu())
        scores = torch.cat(scores)
        return scores.topk(k).indices.tolist()  # 返回Top-K索引

    def _merge_new_samples(self, original_train, new_samples):
        """将新标注的样本合并到训练集"""
        new_df = pd.DataFrame(new_samples)
        merged_df = pd.concat([original_train['df'], new_df], ignore_index=True)
        return {'wild_type': original_train['wild_type'], 'df': merged_df}

    def finetune_single(self, train, valid, save_dir=None):
        args = self.args
        model, tokenizer = self.get_base_model()
        round, labeled_count, train_loss, max_activity, ndcg, topk = [],[],[],[],[],[]
        # 1. 初始化LoRA（如果启用）
        if args.lora_r > 0:
            lora_config = LoraConfig( # esm1b_t33_650M_UR50S
                r=args.lora_r, # 8
                lora_alpha=32, #16
                target_modules=["query","key","value","dense"],
                lora_dropout=0.05,
                bias='none',
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, lora_config).to(self.device)
            print_trainable_params(model)

        # 2. 初始训练集（随机选择12个样本）
        # 确保使用位置索引
        train_df = train['df'].reset_index(drop=True)
        # 初始采样
        initial_size = 12
        # initial_indices = np.random.choice(len(train_df), initial_size, replace=False)
        initial_indices = np.arange(initial_size)
        initial_train = {
            'wild_type': train['wild_type'],
            'df': train_df.iloc[initial_indices].copy()
        }
        remaining_train = {
            'wild_type': train['wild_type'],
            'df': train_df.drop(initial_indices).copy()
        }
        print(initial_train)
        # 3. 主动学习循环
        for al_round in range(args.max_al_rounds):  #最大主动学习轮次
            round.append(al_round + 1)
            labeled_count.append(len(initial_train['df']))
            print(f"\n=== Active Learning Round {al_round + 1}/{args.max_al_rounds}  ===")
            # 生成偏好对（从第二轮开始）
            pref_pairs = None
            if al_round > 0 and args.pref_batch_size > 0:
                pref_pairs = generate_pref_pairs_from_scores(
                    torch.tensor(initial_train['df']['DMS_score'].values, device=self.device),
                    batch_size=args.pref_batch_size,
                    top_score_ratio=0.1,
                    current_epoch=0,
                    total_epochs=args.max_al_rounds-1
                )

            # 3.1 训练当前模型
            train_data = RankingSequenceData(
                initial_train, tokenizer,
                mask=args.mask in {'train', 'all'},
                list_size=args.list_size,
                max_size=args.max_iter * args.train_batch,
                constructor=self.data_constructor,
                device=self.device,
                pref_pairs=pref_pairs
            )
            train_loader = DataLoader(
                train_data,
                batch_size=args.train_batch,
                shuffle=True,
                collate_fn=train_data.collate
            )
            trainer = RankingTrainer(
                model,
                optimizer=args.optimizer,
                lr=args.learning_rate,
                epochs=args.epochs_per_al_round,  # 每轮主动学习的epoch数
                max_grad_norm=args.max_grad_norm,
                score_fn=self.score_fn,
                log_metrics=metrics,
                save_dir=save_dir,
                patience=args.patience,
                bt_weight=args.bt_weight# 渐进增强  * (al_round / args.max_al_rounds)
            )

            if al_round < args.max_al_rounds -1:
                logs = trainer(train_loader, False)  # 仅训练，不保存
            else:
                logs = trainer(train_loader, True) #保存参数

            train_loss.append(logs['train_loss'])
            # 3.2 选择Top-K未标注样本（模拟人类标注）
            topk_indices = self._get_topk_samples(model, remaining_train, tokenizer, k=args.train_size)
            new_samples = remaining_train['df'].iloc[topk_indices].copy()
            # 3.3 更新训练集和未标注集
            initial_train = self._merge_new_samples(initial_train, new_samples)
            remaining_train['df'] = remaining_train['df'].drop(remaining_train['df'].index[topk_indices])
            # 3.4 检查终止条件（如未标注集为空或性能收敛）
            if len(remaining_train['df']) == 0:
                print("All samples labeled. Stopping...")
                break
            # 4. 最终评估（可选）
            if valid is not None:
                eval_data = self.data_constructor(remaining_train, tokenizer, mask=args.mask in {'eval', 'all'}, device=self.device)
                eval_loader = DataLoader(eval_data, batch_size=args.eval_batch, collate_fn=eval_data.collate)
                _, final_metrics = trainer.evaluate_epoch(eval_loader)
                max_activity.append(final_metrics['max_activity'])
                ndcg.append(final_metrics['ndcg'])
                topk.append(final_metrics['topk_pr'])

        self.report['round'] = round
        self.report['labeled_count'] = labeled_count
        self.report['train_loss'] = train_loss
        self.report['max_activity'] = max_activity
        self.report['ndcg'] = ndcg
        self.report['topk'] = topk

        return self.report

    def run_training_and_save_history(self, train, test=None):
        args = self.args
        save_dir = self.get_save_dir(args.mode, train['name'])

        # 执行主动学习循环
        al_history = self.finetune_single(train, test, save_dir)

        # 创建可视化文件夹（保留原有逻辑）
        fig_base_dir = '/public/home/yunjiong/Pycharm_workplace/EvoProtein/result/{}_7kmb_esm2-3B'.format(datetime.datetime.now().strftime("%Y-%m-%d"))
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_folder = os.path.join(fig_base_dir, f"{train['name']}_{timestamp}_{self.seed}")
        os.makedirs(output_folder, exist_ok=True)

        df_history = pd.DataFrame(al_history)
        csv_path = os.path.join(output_folder, 'al_history.csv')
        df_history.to_csv(csv_path, index=False)
        print(f"Saved active learning history to {csv_path}")
        # 绘制可视化图形
        # self.plot_al_performance(al_history,  output_folder, train['name'])

        return al_history

    def plot_al_performance(self, al_history, output_folder, protein_name):
        """
        绘制主动学习性能变化图并保存

        参数:
            al_history: 包含训练历史的字典
            output_folder: 图片保存路径
            protein_name: 蛋白质名称(用于标题)
        """
        plt.figure(figsize=(12, 6))

        # 获取数据
        rounds = al_history['labeled_count']
        max_activity = al_history['max_activity']
        ndcg = al_history['ndcg']
        topk = al_history['topk']

        # 创建折线图
        line1, = plt.plot(rounds, max_activity, 'b-o', label='max_activity')
        line2, = plt.plot(rounds, ndcg, 'g--s', label='NDCG')
        line3, = plt.plot(rounds, topk, 'r-.D', label='Top-K Accuracy')

        # 设置图表元素
        plt.title(f'Model  Performance During Active Learning - {protein_name}')
        plt.xlabel('Active  Learning Round')
        plt.ylabel('Score')
        plt.legend(handles=[line1, line2, line3], loc='lower right')
        plt.grid(True, alpha=0.3)

        # 自动调整y轴范围
        min_val = min(min(max_activity), min(ndcg), min(topk))
        max_val = max(max(max_activity), max(ndcg), max(topk))
        plt.ylim(max(min_val - 0.1, 0), min(max_val + 0.1, 1))

        # 添加数据标签
        for i, (s, n, t) in enumerate(zip(max_activity, ndcg, topk)):
            plt.text(rounds[i], s, f'{s:.2f}', ha='center', va='bottom')
            plt.text(rounds[i], n, f'{n:.2f}', ha='center', va='top')
            plt.text(rounds[i], t, f'{t:.2f}', ha='center', va='bottom')

        # 保存图表
        plot_path = os.path.join(output_folder, 'al_performance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved performance plot to {plot_path}")


    def test_single(self, train, test):
        args = self.args
        if args.epochs > 0:
            load_dir = self.get_save_dir(args.mode, test['name'])
            if args.lora_r == 0:
                model, tokenizer = self.get_base_model(load_dir)
            else:
                model, tokenizer = self.get_base_model()
                model = PeftModel.from_pretrained(model, load_dir, is_trainable=True)
        else:
            model, tokenizer = self.get_base_model()
        
        test_data = self.data_constructor(test, tokenizer,
                                          mask=args.mask in {'eval', 'all'},
                                          device=self.device)
        test_iter = DataLoader(test_data,
                               batch_size=args.eval_batch,
                               collate_fn=test_data.collate)
        trainer = RankingTrainer(model.to(self.device), log_metrics=[], score_fn=self.score_fn)
        predicts, _ = trainer.evaluate_epoch(test_iter)
        predicts = predicts.tolist()
        
        predicts = pd.Series(predicts, index=test['df'].index, name='prediction')
        report, _ = group_scores(train['df'], predicts, test['df'])
        print('======================Breakdown results======================')
        print(report)
        
        print('Saving model predictions...')
        save_path = self.get_save_dir(args.mode, test['name'], prediction=True)
        save_path += '_base.csv' if args.epochs == 0 else '.csv'
        make_dir(save_path)
        predicts.to_csv(save_path)
        return report
    
    def select_datasets(self, all_proteins):
        args = self.args

        if args.protein in all_proteins.keys():
            return all_proteins[args.protein]
        
        proteins = chain(*all_proteins.values())
        if args.train_size >= 1:
            proteins = filter(lambda x: len(x['df']) > args.train_size, proteins)
        
        if args.protein == 'all':
            return list(proteins)
        if args.protein == 'single-site':
            return list(filter(lambda x: x['n_sites'][-1] == 1, proteins))
        if args.protein == 'multi-site':
            return list(filter(lambda x: x['n_sites'][-1] > 1, proteins))
        if len(args.protein) == 2:
            proteins = list(proteins)
            N, i = int(args.protein[0]), int(args.protein[1])
            n = ceil(len(proteins) / N)
            j = (i - 1) * n
            return proteins[j:j + n]
    
    def __call__(self, all_proteins):
        args = self.args
        proteins = self.select_datasets(all_proteins)
        reports = {}
        for protein in proteins:
            print(f'**********************Current dataset: {protein["name"]}**********************')
            if protein['name'] == 'CCDB_ECOLI_Tripathi_2016':
                eval_metric = args.eval_metric
                args.eval_metric = 'ndcg' # in case of nan spearmanr
            
            train, test = split_data(protein, args.train_data_size, n_sites=args.n_sites,
                                     neg_train=args.negative_train, scale=False)
            if args.test:
                report = self.test_single(train, test)
            else:
                report = self.run_training_and_save_history(train, test)
            reports[protein['name']] = report
            torch.cuda.empty_cache()

            if protein['name'] == 'CCDB_ECOLI_Tripathi_2016':
                args.eval_metric = eval_metric

        
        if args.test and args.protein in {'single-site', 'multi-site', 'all'}:
            save_path = self.get_save_dir(args.mode, args.protein, prediction=True)
            save_path += '_base.pkl' if args.epochs == 0 else '.pkl'
            make_dir(save_path)
            reports = summarize_scores(reports, save_path)
            print('**********************Score summary**********************')
            print(reports[args.eval_metric])

        return max(self.report['max_activity'])