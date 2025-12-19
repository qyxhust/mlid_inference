import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# ----------------- 1. 读取 + 对齐 -----------------
pred = pd.read_csv("data/em/em_results.csv")    # 你第一张图对应的文件
true = pd.read_csv("data/simulate/labels.tsv", sep="\t")  # 你第二张图对应的文件

pred.columns = pred.columns.str.strip()
true.columns = true.columns.str.strip()

# 如果列名不是 sample / population，在这里改一下：
# pred.rename(columns={"你的样本列名": "sample"}, inplace=True)
# true.rename(columns={"你的样本列名": "sample",
#                      "你的标签列名": "population"}, inplace=True)

df = pred.merge(true, on="sample", how="inner")

cls = ["CEU", "Neandertal", "YRI"]      # 类别顺序
df["pred_label"] = df[cls].idxmax(axis=1)
df["true_label"] = df["population"]

cm = confusion_matrix(df["true_label"], df["pred_label"], labels=cls)
cm_df = pd.DataFrame(cm, index=cls, columns=cls)
print(cm_df)

# ----------------- 2. 画混淆矩阵图 -----------------
fig, ax = plt.subplots(figsize=(5, 4))

im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Count")

# 坐标轴刻度和标签
ax.set_xticks(np.arange(len(cls)))
ax.set_yticks(np.arange(len(cls)))
ax.set_xticklabels(cls)
ax.set_yticklabels(cls)

ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Confusion Matrix")

# 在每个格子上写数字
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j, i, cm[i, j],
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=9
        )

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()
