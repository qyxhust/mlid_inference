import pandas as pd

# 1. fam 文件：提供样本顺序
fam = pd.read_csv("truth.pruned.fam", sep=r"\s+", header=None)
fam.columns = ["FID", "IID", "pid", "mid", "sex", "pheno"]

# 2. 所有样本的真实 population 信息
all_labels = pd.read_csv("data/simulate/labels.tsv", sep="\t")   # 改成你的路径
all_labels = all_labels.set_index("sample")        # index=sample 名字

# 3. 测试集（25%）样本列表
test = pd.read_csv("data/test_samples.csv")  # 改成你的路径
test_ids = set(test["sample"])                     # 出现在这里的样本视为“测试：不监督”

pop_lines = []

for iid in fam["IID"]:
    if iid in test_ids:
        # 这 25%：不给 ADMIXTURE 标签，用 "-" 让它自己推断
        pop_lines.append("-")
    else:
        # 其余 75%：用真实祖源标签做 supervised 参考
        pop_lines.append(all_labels.loc[iid, "population"])

# 4. 写出 .pop 文件
with open("truth.pruned.pop", "w") as f:
    for lab in pop_lines:
        f.write(str(lab) + "\n")

print("写好了 truth.pruned.pop，行数 =", len(pop_lines))

