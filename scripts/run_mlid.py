import yaml
import sys
import pandas as pd
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mlid import TaxidentRunner, EMRunner
from src.align import EMAlignment # Note: pipeline.py imported from scripts.run_align but src.align is likely better if it exists there. 
# pipeline.py had: from scripts.run_align import EMAlignment
# scripts/run_mlid.py had: from src.align import EMAlignment
# I will trust the existing import in scripts/run_mlid.py
from src.analysis import AncestryAnalysis

def process_one_sample(
    sample_id: str,
    aligner: EMAlignment,
    tax_runner: TaxidentRunner,
    em_runner: EMRunner,
    bowtie2_path: str = "bowtie2",
    bowtie2_build: str = "bowtie2-build",
    samtools_path: str = "samtools",
    threads: int = 4,
):
    print("\n============================")
    print(f"[PIPELINE] 处理样本: {sample_id}")

    # 1. 优先检查结果是否已存在 (将此步骤移到最前，避免不必要的索引检查)
    # 注意：现在 em_runner._sample_in_result 默认检查的是 em_results.csv (hap-level)
    # 同时也尝试检查 final_result.csv (pop-level)
    if em_runner._sample_in_result(sample_id):
        print(f"[INFO] sample={sample_id} 已在结果中，跳过处理")
        return

    # 0. 确保索引（多次调用也只会在第一次真正建）
    aligner.build_bowtie2_index(bowtie2_build=bowtie2_build)

    # 2. 检查 TaxIdent 输出是否已经存在
    taxident_path = tax_runner.taxident_output_path(sample_id)
    if taxident_path.exists():
        print(f"[INFO] 发现已有 TaxIdent 输出，将从 EM 步骤继续: {taxident_path}")
        em_runner.run_em_if_needed(sample_id, taxident_path)
        return

    # 3. 检查 BAM 是否存在
    bam_path = aligner.align_dir / f"{sample_id}.sorted.bam"
    if bam_path.exists():
        print(f"[INFO] 发现已有 BAM，将从 TaxIdent 步骤继续: {bam_path}")
        # 保持 BAM 文件不删除
        taxident_out = tax_runner.ensure_taxident(
            sample_id, bam_path, delete_bam_after=False, threads=threads
        )
        em_runner.run_em_if_needed(sample_id, taxident_out)
        return

    # 4. 上面都没有 -> 只能从 reads 开始跑完整流程
    print("[INFO] 未发现 BAM 或 TaxIdent 输出，将从对齐开始")
    bam_path = aligner.ensure_bam(
        sample_id,
        bowtie2_path=bowtie2_path,
        samtools_path=samtools_path,
        threads=threads,
        delete_reads_after=False, # Changed to False to preserve reads for other pipelines
    )
    # 保持 BAM 文件不删除
    taxident_out = tax_runner.ensure_taxident(
        sample_id, bam_path, delete_bam_after=False, threads=threads
    )
    em_runner.run_em_if_needed(sample_id, taxident_out)

def run_mlid_pipeline(model_type="hard"):
    """
    运行 MLID (TaxIdent + EM) 流程
    """
    cfg = yaml.safe_load(Path("config/default.yaml").read_text())
    
    print("\n[PIPELINE] 启动 MLID 分支...")
    
    test_samples_csv = cfg["project"]["test_samples"]
    ref_samples_csv = cfg["project"]["ref_samples"]
    align_root = cfg["project"]["align_root"]
    reads_root = cfg["project"]["reads_root"]
    ref_fasta_path = cfg["project"]["ref_haps"]
    
    # Run Name
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    depth = cfg["reads"].get("depth", 1)
    algo = "mlid"
    # Format: model_algorithm_depth_timestamp
    run_name = f"{model_type}_{algo}_{depth}x_{timestamp}"
    
    result_base = cfg["project"].get("result_base", "results")
    run_dir = ROOT / result_base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] MLID 结果目录: {run_dir}")

    central_final_csv = cfg["project"]["final_result_csv"]
    central_em_csv = str(Path(central_final_csv).parent / "em_results.csv")
    align_out_root = cfg["project"]["align_root"]
    
    # Tools
    taxident_path = str(ROOT / "tools/TaxIdent/TaxIdent")
    bowtie2_build = "bowtie2-build"
    bowtie2_path = "bowtie2"
    samtools_path = "samtools"
    threads = 8

    # 1. 检查是否全跑完
    if Path(central_final_csv).exists():
        # 这里省略复杂的全样本检查，假设若文件很新则已更新
        pass

    if not Path(test_samples_csv).exists():
        print("[ERROR] 测试样本列表不存在，无法运行 MLID。请先运行 data 步骤。")
        return

    test_df = pd.read_csv(test_samples_csv)
    sample_ids = test_df["sample"].astype(str).tolist()

    # 2. 初始化组件
    aligner = EMAlignment(ref_fasta_path, reads_root, align_out_root)
    # 确保索引
    if not Path(f"{ref_fasta_path}.1.bt2").exists():
        aligner.build_bowtie2_index(bowtie2_build=bowtie2_build)
    
    hap_index_df = aligner.write_hap_index_table()
    tax_runner = TaxidentRunner(taxident_path, align_out_root)
    em_runner = EMRunner(hap_index_df, align_out_root, central_em_csv, final_result_csv=central_final_csv)

    # 3. 循环处理
    print(f"[INFO] 开始处理 {len(sample_ids)} 个样本...")
    for sid in sample_ids:
        process_one_sample(
            sample_id=sid,
            aligner=aligner,
            tax_runner=tax_runner,
            em_runner=em_runner,
            bowtie2_path=bowtie2_path,
            bowtie2_build=bowtie2_build,
            samtools_path=samtools_path,
            threads=threads,
        )
        em_runner.aggregate_to_population(ref_samples_csv, central_final_csv)

    print("[OK] MLID 计算流程完成。")

    # 4. 分析绘图
    print(f"[ANALYSIS] MLID 分析与绘图...")
    run_final_csv = run_dir / "final_result.csv"
    run_em_csv = run_dir / "em_results.csv"
    
    if Path(central_final_csv).exists():
        shutil.copy2(central_final_csv, run_final_csv)
    if Path(central_em_csv).exists():
        shutil.copy2(central_em_csv, run_em_csv)
        
    truth_csv_path = test_samples_csv 
    if run_final_csv.exists():
        ana = AncestryAnalysis(str(run_final_csv), truth_csv_path, str(run_dir))
        
        if model_type == "admix":
            try:
                ana.plot_admixture_barplot(top_n=100)
            except Exception as e:
                print(f"[WARN] Plotting failed: {e}")
        else:
            try:
                ana.plot_confusion_matrix()
            except Exception as e:
                print(f"[WARN] Plotting failed: {e}")
    else:
        print("[WARN] 无结果文件，跳过绘图。")

if __name__ == "__main__":
    run_mlid_pipeline()
