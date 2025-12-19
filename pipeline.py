# pipeline.py —— 中央调度器
import argparse
import sys
import yaml
import pandas as pd
import shutil
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

# 导入各模块功能
from scripts.run_simulate import run_simulate
from scripts.run_read import run_read
from src.read import split_samples_dual_mode
from scripts.run_align import EMAlignment
from src.mlid import TaxidentRunner, EMRunner
from scripts.run_mlid import run_mlid_pipeline
from src.analysis import AncestryAnalysis
from scripts.recover_reference import recover_reference
from scripts.run_admixture import run_admixture_pipeline
from scripts.run_ohana import run_ohana_pipeline


def ensure_fasta_indexes(ref_path: Path):
    """
    确保参考序列的索引是最新的：
    - samtools faidx (.fai)
    - GATK sequence dictionary (.dict)
    """
    if not ref_path.exists():
        return

    ref_fai = ref_path.with_suffix(ref_path.suffix + ".fai")
    ref_dict = ref_path.with_suffix(".dict")

    # faidx: 若不存在或比 fasta 旧则重建
    if (not ref_fai.exists()) or (ref_fai.stat().st_mtime < ref_path.stat().st_mtime):
        print(f"[INFO] (faidx) 重新生成 {ref_fai} ...")
        subprocess.run(["samtools", "faidx", str(ref_path)], check=True)

    # dict: 同理
    if (not ref_dict.exists()) or (ref_dict.stat().st_mtime < ref_path.stat().st_mtime):
        print(f"[INFO] (dict) 重新生成 {ref_dict} ...")
        gatk_bin = Path(__file__).resolve().parent / "tools/gatk/gatk"
        subprocess.run([str(gatk_bin), "CreateSequenceDictionary", "-R", str(ref_path), "-O", str(ref_dict)], check=True)

def ensure_data_ready(cfg, model_type="hard"):
    """
    检查并生成公共基础数据:
    1. Truth VCF (msprime)
    2. Sample Split (Ref/Test lists)
    3. Reads (wgsim)
    4. Reference Genome (reference.fa)
    """
    print("\n[PIPELINE] 正在检查基础数据完整性 (Step: data)...")
    
    # 1. Check VCF
    vcf_path = Path(cfg["project"]["simulate_vcfs"])
    if not vcf_path.exists() or vcf_path.stat().st_size == 0:
        print(f"[INFO] VCF 未找到或为空，开始运行模拟 (Mode: {model_type})...")
        run_simulate(model_type=model_type)
    else:
        print(f"[OK] VCF 已存在: {vcf_path}")

    # 2. Check Sample Split
    ref_samples_csv = Path(cfg["project"]["ref_samples"])
    test_samples_csv = Path(cfg["project"]["test_samples"])
    labels_path = cfg["project"]["simulate_label"]
    
    # 只有当 labels 文件存在时才能 split
    if Path(labels_path).exists():
        if not ref_samples_csv.exists() or not test_samples_csv.exists():
            print("[INFO] 样本划分文件缺失，正在划分 Ref/Test...")
            split_samples_dual_mode(
                meta_path=labels_path,
                ref_out=str(ref_samples_csv),
                test_out=str(test_samples_csv),
                mode=model_type,
                ref_ratio=0.75,
                seed=42
            )
        else:
            print("[OK] 样本划分文件已存在。")
    else:
        print("[WARN] 标签文件不存在，跳过样本划分 (可能模拟刚失败)。")

    # 3. Check Reads
    reads_root = Path(cfg["project"]["reads_root"])
    # 简单检查：如果 reads 目录不存在或为空，则运行 run_read
    if not reads_root.exists() or not any(reads_root.iterdir()):
        print("[INFO] Reads 目录为空或缺失，正在生成 Reads...")
        run_read()
    else:
        # 也可以更细致地检查 ref_haps.fa
        ref_haps = Path(cfg["project"]["ref_haps"])
        if not ref_haps.exists():
             print("[INFO] ref_haps.fa 缺失，重新运行 run_read 以生成...")
             run_read()
        else:
             print("[OK] Reads 与 Reference Haplotypes 已存在。")

    # 4. Check Recovered Reference (for GATK/ANGSD)
    data_root = Path(cfg["project"].get("data_root", "/space/s1/qyx/data"))
    ref_fa = Path(cfg["project"].get("ref_genome", data_root / "reference.fa"))
    if not ref_fa.exists():
        print("[INFO] reference.fa (单一参考) 未找到，正在从 VCF 恢复...")
        recover_reference()
    else:
        print(f"[OK] reference.fa 已存在: {ref_fa}")
        ensure_fasta_indexes(ref_fa)

def main():
    parser = argparse.ArgumentParser(description="Ancestry Inference Pipeline Manager")
    parser.add_argument("--steps", type=str, default="data,mlid", 
                        help="Comma-separated steps: data, mlid, admixture, ohana. Default: data,mlid")
    parser.add_argument("--model", type=str, default="normal",
                        help="Simulation model type: hard, normal, admix. Default: hard")
    parser.add_argument("--unsupervised", action="store_true", 
                        help="Run Ohana in unsupervised mode (default: supervised)")
    
    args = parser.parse_args()
    steps = [s.strip().lower() for s in args.steps.split(",")]
    model_type = args.model
    supervised = not args.unsupervised

    print(f"==================================================")
    print(f"Pipeline Started. Steps: {steps}, Model: {model_type}, Supervised: {supervised}")
    print(f"==================================================")

    cfg = yaml.safe_load(Path("config/default.yaml").read_text())
    
    # Step 1: Data
    if "data" in steps:
        ensure_data_ready(cfg, model_type=model_type)
    
    # Step 2: MLID
    if "mlid" in steps:
        # 确保参考索引最新（即使未跑 data 步骤）
        data_root = Path(cfg["project"].get("data_root", "/space/s1/qyx/data"))
        ref_fa_mlid = Path(cfg["project"].get("ref_haps", data_root / "ref_haps.fa"))
        ensure_fasta_indexes(ref_fa_mlid)
        run_mlid_pipeline(model_type=model_type)
        time.sleep(1) # Ensure timestamp difference if multiple steps run quickly
        
    # Step 3: GATK + Admixture
    if "admixture" in steps:
        print("\n[PIPELINE] 启动 GATK + Admixture 分支...")
        data_root = Path(cfg["project"].get("data_root", "/space/s1/qyx/data"))
        ref_fa_adm = Path(cfg["project"].get("ref_genome", data_root / "reference.fa"))
        ensure_fasta_indexes(ref_fa_adm)
        # Admixture 脚本内部有自己的 Config 读取和流程控制
        # 我们假设它能自己处理好，或者我们可以传递一些参数
        # 目前直接调用封装好的函数
        run_admixture_pipeline(model_type=model_type)
        time.sleep(1)

    # Step 4: ANGSD + Ohana
    if "ohana" in steps:
        print("\n[PIPELINE] 启动 ANGSD + Ohana 分支...")
        data_root = Path(cfg["project"].get("data_root", "/space/s1/qyx/data"))
        ref_fa_ohana = Path(cfg["project"].get("ref_genome", data_root / "reference.fa"))
        ensure_fasta_indexes(ref_fa_ohana)
        run_ohana_pipeline(model_type=model_type, supervised=supervised)

    print("\n[DONE] All requested steps completed.")

if __name__ == "__main__":
    main()
