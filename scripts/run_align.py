from src.align import EMAlignment
import yaml
from pathlib import Path

def run_align():
    cfg = yaml.safe_load(Path("config/default.yaml").read_text())

    ref_fasta = cfg["project"]["ref_haps"]
    test_samples_csv = cfg["project"]["test_samples"]
    reads_root = cfg["project"]["reads"]
    align_root = cfg["project"]["align"]
    # 75% 参考 hap 的 multi-FASTA

    aligner = EMAlignment(
        ref_fasta_path=ref_fasta,
        test_samples_csv=test_samples_csv,
        reads_root=reads_root,
        out_root=align_root,
    )

    # 1. 建 bowtie2 索引 + hap_index 表
    # 检查 index 是否已存在，如果已存在则跳过
    index_prefix = str(Path(align_root) / "ref_haps")
    if not Path(index_prefix + ".1.bt2").exists():
        aligner.build_bowtie2_index(bowtie2_build="bowtie2-build")
    else:
        print(f"[INFO] Bowtie2 index already exists at {index_prefix}.*.bt2, skipping build.")
    hap_index_df = aligner.write_hap_index_table()  # data/em/ref_haps_index.csv

    # 2. 对所有 test 样本做对齐，生成 sorted BAM
    aligner.align_all_samples(
        bowtie2_path="bowtie2",
        samtools_path="samtools",
        threads=8,
    )

    # 3. 从 BAM 生成 EM 需要的 read–hap mismatch TSV
    aligner.em_input_all_samples(hap_index_df=hap_index_df)
