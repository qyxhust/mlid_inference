# scripts/run_read.py
import gzip
import yaml
from pathlib import Path
from typing import List

from src.read import SequencingRead  # 负责：从 VCF 构建 haplotypes、写 FASTA


def run_read():
    cfg = yaml.safe_load(Path("config/default.yaml").read_text())
    
    seq = SequencingRead(
    vcf_path=str(cfg["project"]["simulate_vcfs"]),
    L=cfg["msprime"]["l"],
    read_root=str(cfg["project"]["reads_root"]),
    chrom="1",
    test_table="/space/s1/qyx/data/test_samples.csv",   # 25% 列表（sample_id + population）
)

    # 1) 生成 75% reference 的总 FASTA（EM 参考库）
    seq.write_reference_haplotypes_merged(out_fa="/space/s1/qyx/data/ref_haps.fa")

    # 2) 为 25% test 样本写各自的 fasta
    seq.write_test_haplotypes_per_sample()

    # 3) 对 25% test 样本跑双端 wgsim
    seq.run_wgsim_for_tests(depth=cfg["reads"]["depth"])

    # 4) [DISABLED] 对 75% reference 样本跑高深度 wgsim
    # 我们采用 VCF -> GL 直接转换方案，不再需要生成参考样本的 Reads
    # seq.run_wgsim_for_refs(depth=30.0)

    print("[OK] Read simulation complete.")


if __name__ == "__main__":
    run_read()
