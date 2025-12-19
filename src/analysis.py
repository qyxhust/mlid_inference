import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shutil
import numpy as np

class AncestryAnalysis:
    def __init__(self, result_csv_path, truth_csv_path, output_dir, test_samples_path=None):
        """
        result_csv_path: 您的 pipeline 生成的 final_result.csv (包含预测比例)
        truth_csv_path: 模拟时生成的真实比例文件 (ref_samples.csv 或 test_samples.csv 里的真实标签)
        output_dir: 结果输出和图片保存的目录
        test_samples_path: (Optional) Path to test_samples.csv. 
                           If provided, plotting can be restricted to these samples only.
        """
        self.result_csv = Path(result_csv_path)
        self.truth_csv = Path(truth_csv_path)
        self.out_dir = Path(output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        self.df_pred = pd.read_csv(self.result_csv)
        
        # Try reading truth csv with tab separator first, then comma
        try:
            self.df_truth = pd.read_csv(self.truth_csv, sep='\t')
            if 'sample' not in self.df_truth.columns:
                 self.df_truth = pd.read_csv(self.truth_csv, sep=',')
        except:
             self.df_truth = pd.read_csv(self.truth_csv, sep=None, engine='python')
        
        # 确保 sample 列是字符串以进行合并
        self.df_pred['sample'] = self.df_pred['sample'].astype(str)
        self.df_truth['sample'] = self.df_truth['sample'].astype(str)

        # Load test samples filter if provided
        self.test_samples = None
        if test_samples_path and Path(test_samples_path).exists():
            try:
                df_test = pd.read_csv(test_samples_path)
                if 'sample' in df_test.columns:
                    self.test_samples = set(df_test['sample'].astype(str))
                    print(f"[INFO] Loaded {len(self.test_samples)} test samples for filtering plots.")
            except Exception as e:
                print(f"[WARN] Failed to load test samples from {test_samples_path}: {e}")

        # 合并预测和真实数据
        # 假设 df_truth 包含 'sample' 和 'pop' (真实群体/标签)
        # 如果是 admix 模式，truth 文件可能需要包含真实的混合比例 (例如 Q_popA, Q_popB...)
        # 这里先做一个通用的 merge
        self.merged_df = pd.merge(self.df_truth, self.df_pred, on='sample', how='inner')

    def archive_results(self, run_name):
        """
        将当前运行的结果归档到一个指定的子文件夹
        """
        archive_dir = self.out_dir / "archive" / run_name
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制结果 CSV
        shutil.copy2(self.result_csv, archive_dir / self.result_csv.name)
        print(f"[INFO] Results archived to {archive_dir}")
        return archive_dir

    def plot_confusion_matrix(self, true_label_col=None):
        """
        仅适用于 'classify' 模式 (每个样本属于单一群体)
        """
        # 自动检测真实标签列名: 'pop' 或 'population'
        if true_label_col is None:
            if 'pop' in self.merged_df.columns:
                true_label_col = 'pop'
            elif 'population' in self.merged_df.columns:
                true_label_col = 'population'
            else:
                raise KeyError("Cannot find 'pop' or 'population' column in merged dataframe.")

        # Filter for plotting
        # Apply filtering for confusion matrix too if test_samples provided
        plot_df = self.merged_df
        if self.test_samples:
            # Only keep samples in self.test_samples
            plot_df = self.merged_df[self.merged_df['sample'].isin(self.test_samples)].copy()
            print(f"[INFO] Filtering confusion matrix plot to {len(plot_df)} test samples.")
        else:
            plot_df = self.merged_df.copy()

        # 找到预测列: 以 Q_ 开头的列
        pred_cols = [c for c in self.df_pred.columns if c.startswith("Q_")]
        
        # 对每个样本，找到预测概率最大的群体
        # 假设列名是 "Q_PopA"，我们需要提取 "PopA"
        # idxmax 会返回列名，我们去掉 "Q_" 前缀
        plot_df['predicted_pop'] = plot_df[pred_cols].idxmax(axis=1).apply(lambda x: x.replace('Q_', ''))
        
        # 计算混淆矩阵
        from sklearn.metrics import confusion_matrix
        
        y_true = plot_df[true_label_col]
        y_pred = plot_df['predicted_pop']
        
        labels = sorted(list(set(y_true) | set(y_pred)))
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # 计算准确率
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_true, y_pred)
        print(f"[INFO] Classification Accuracy: {acc:.4f}")

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix (Accuracy: {acc:.2%})')
        
        out_path = self.out_dir / "confusion_matrix.png"
        plt.savefig(out_path)
        plt.close()
        print(f"[INFO] Confusion matrix saved to {out_path}")

    def plot_admixture_barplot(self, top_n=50, plot_truth=True, only_test=True):
        """
        适用于 'admix' 模式。
        绘制堆叠柱状图 (Stacked Bar Plot)。
        如果 plot_truth=True，且检测到样本属于 'ADMIX' 群体，
        则会尝试绘制真实值与预测值的对比图。
        
        only_test: If True and test_samples are loaded, only plot those samples.
        """
        # 找到所有群体列
        q_cols = sorted([c for c in self.merged_df.columns if c.startswith("Q_")])
        
        # 检查是否为 Admixture 模式 (通过 population 列是否有 'ADMIX' 或者 'admix' 样本)
        # 注意: 这里的 pop 列名可能是 'population' 或 'pop'
        pop_col = 'pop' if 'pop' in self.merged_df.columns else 'population'
        
        is_admix_mode = False
        if pop_col in self.merged_df.columns:
            if self.merged_df[pop_col].astype(str).str.contains("admix|ADMIX", case=False).any():
                is_admix_mode = True

        # Filter for plotting
        if only_test and self.test_samples:
            # Only keep samples in self.test_samples
            plot_df = self.merged_df[self.merged_df['sample'].isin(self.test_samples)].copy()
            print(f"[INFO] Filtering plot to {len(plot_df)} test samples.")
        else:
            plot_df = self.merged_df.copy()
        
        if plot_df.empty:
            print("[WARN] No samples to plot after filtering!")
            return

        # 为了美观，通常按照主要成分排序
        # 找到每个样本最大的成分
        plot_df['max_component'] = plot_df[q_cols].idxmax(axis=1)
        plot_df['max_value'] = plot_df[q_cols].max(axis=1)
        
        # 先按主要成分的名字排序，再按比例大小排序
        plot_df = plot_df.sort_values(by=['max_component', 'max_value'], ascending=[True, False])
        
        if len(plot_df) > top_n:
             print(f"[INFO] Sample count ({len(plot_df)}) > {top_n}. Plotting top {top_n} samples only.")
             plot_df = plot_df.head(top_n)

        samples = plot_df['sample']
        
        # 准备绘图
        # 如果需要画 Truth，我们使用两个子图
        if plot_truth and is_admix_mode:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            axes = [ax1, ax2]
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
            axes = [ax1]

        # 获取颜色调色板
        colors = sns.color_palette("tab20", len(q_cols))
        color_map = {col.replace("Q_", ""): colors[i] for i, col in enumerate(q_cols)}

        # === 1. Plot Predicted ===
        bottom = np.zeros(len(plot_df))
        for i, col in enumerate(q_cols):
            pop_name = col.replace("Q_", "")
            values = plot_df[col].values
            ax1.bar(samples, values, bottom=bottom, label=pop_name, color=color_map.get(pop_name, colors[i]), width=1.0)
            bottom += values
            
        ax1.set_ylabel("Predicted Proportion")
        ax1.set_title("Admixture Proportions (Predicted)")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # === 2. Plot Truth (Optional) ===
        if plot_truth and is_admix_mode and len(axes) > 1:
            ax2 = axes[1]
            bottom_truth = np.zeros(len(plot_df))
            
            # 构造 Truth Data
            # 规则: 
            # - 如果 pop == ADMIX: AFR=1/6, EUR=1/3, ASIA=1/2 (基于用户提供的 AmericanAdmixture_4B18 信息)
            # - 如果 pop == AFR: AFR=1.0
            # - 如果 pop == EUR: EUR=1.0
            # - 如果 pop == ASIA: ASIA=1.0
            
            # 硬编码的 Truth 比例 (按列名排序)
            # 假设 q_cols 包含 Q_AFR, Q_ASIA, Q_EUR (排序后)
            
            # 预先计算每一列的 Truth 值
            truth_values = {col: [] for col in q_cols}
            
            for _, row in plot_df.iterrows():
                p_name = row[pop_col]
                
                # 默认为 0
                row_truth = {col: 0.0 for col in q_cols}
                
                if "ADMIX" in str(p_name).upper():
                    # 混合群体
                    # 映射: Q_AFR -> 1/6, Q_EUR -> 1/3, Q_ASIA -> 1/2
                    # 需要确保列名匹配。假设列名包含这些关键字
                    for col in q_cols:
                        c_upper = col.upper()
                        if "AFR" in c_upper:
                            row_truth[col] = 1/6
                        elif "EUR" in c_upper:
                            row_truth[col] = 1/3
                        elif "ASIA" in c_upper:
                            row_truth[col] = 1/2
                        # 归一化一下以防万一 (1/6+1/3+1/2 = 1.0)
                else:
                    # 纯群体 (Source Pop)
                    # 找到对应的 Q_ 列
                    # 比如 pop="AFR", 对应 Q_AFR = 1.0
                    matched = False
                    for col in q_cols:
                        if p_name in col: # 简单匹配: pop "AFR" in "Q_AFR"
                            row_truth[col] = 1.0
                            matched = True
                            break
                    if not matched:
                        # 尝试更模糊的匹配
                        for col in q_cols:
                            if p_name in col.replace("Q_", ""): 
                                row_truth[col] = 1.0

                # 存入列表
                for col in q_cols:
                    truth_values[col].append(row_truth[col])

            # 绘制 Truth Stacked Bar
            for i, col in enumerate(q_cols):
                pop_name = col.replace("Q_", "")
                t_vals = np.array(truth_values[col])
                
                # 使用与 Predicted 相同的颜色映射
                bar_color = color_map.get(pop_name, colors[i])
                
                # 绘制 Truth Bar
                ax2.bar(samples, t_vals, bottom=bottom_truth, label=pop_name, color=bar_color, width=1.0)
                bottom_truth += t_vals
                
            ax2.set_ylabel("True Proportion (Expected)")
            ax2.set_title("Admixture Proportions (Ground Truth)")
            ax2.set_xlabel("Samples")
            plt.setp(ax2.get_xticklabels(), rotation=90, fontsize=8)
            # ax2 不需要 legend，因为颜色含义一样
        
        else:
            plt.setp(ax1.get_xticklabels(), rotation=90, fontsize=8)
            ax1.set_xlabel("Samples")

        plt.tight_layout()
        
        out_path = self.out_dir / "admixture_structure_plot.png"
        plt.savefig(out_path)
        plt.close()
        print(f"[INFO] Admixture plot saved to {out_path}")

    def plot_scatter_pie(self):
        """
        如果知道样本的地理位置 (Lat/Lon)，可以在地图上画饼图。
        这里先留个占位符，如果未来有地理数据可以用。
        """
        pass

if __name__ == "__main__":
    # 示例调用
    import sys
    # 简单测试参数
    # python scripts/analysis.py result.csv truth.csv output_dir
    if len(sys.argv) > 3:
        res_path = sys.argv[1]
        truth_path = sys.argv[2]
        out_path = sys.argv[3]
        
        ana = AncestryAnalysis(res_path, truth_path, out_path)
        
        # 自动判断模式？或者都画
        try:
            print("Attempting to plot confusion matrix (assuming classification)...")
            ana.plot_confusion_matrix()
        except Exception as e:
            print(f"Skipping confusion matrix: {e}")
            
        try:
            print("Attempting to plot admixture barplot...")
            ana.plot_admixture_barplot(top_n=100)
        except Exception as e:
            print(f"Skipping admixture plot: {e}")

