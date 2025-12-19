import yaml
import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.admixture import GATKRunner, run_admixture_from_vcf
from src.read import SequencingRead
import subprocess

def run_admixture_pipeline(model_type="hard"):
    # 1. Load Config
    cfg = yaml.safe_load(Path("config/default.yaml").read_text())
    
    # ... (code omitted) ...
    
    # Output naming
    labels_path = cfg["project"]["simulate_label"]
    model_mode = model_type # Use passed argument
    
    # Extract config paths
    ref_fasta = Path(cfg["project"]["ref_genome"])
    reads_root = Path(cfg["project"]["reads_root"])
    align_root = Path(cfg["project"]["align_root"])
        
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    depth = cfg["reads"].get("depth", 1)
    algo = "gatk_admixture"
    
    # Format: model_algorithm_depth_timestamp
    run_name = f"{model_mode}_{algo}_{depth}x_{timestamp}"
    
    result_base = cfg["project"].get("result_base", "results")
    out_dir = Path(result_base) / run_name
    
    # Tool Path
    gatk_path = ROOT / "tools/gatk/gatk"
    
    print(f"[INFO] GATK+Admixture Pipeline Start")
    print(f"       Reference: {ref_fasta}")
    print(f"       Output:    {out_dir}")
    
    runner = GATKRunner(gatk_path, ref_fasta, out_dir, align_root)
    runner.ensure_ref_index()
    
    # Simple Mode: No re-alignment logic needed if we trust the simulation
    # But GATK needs BAMs. We generate BAMs by aligning reads to reference.fa.
    # This IS the standard flow.
    # We remove the complex logic about "use_realigned" vs "ref_haps.fa".
    
    bowtie2_index = str(ref_fasta).rsplit(".", 1)[0]
        if not Path(bowtie2_index + ".1.bt2").exists():
            print("[INFO] Building Bowtie2 index for reference.fa...")
            subprocess.run(["bowtie2-build", str(ref_fasta), bowtie2_index], check=True)

    # 2. Collect Samples
    test_samples_csv = cfg["project"]["test_samples"]
    labels_path = cfg["project"]["simulate_label"]
    
    if Path(test_samples_csv).exists():
        try:
             df_test = pd.read_csv(test_samples_csv, sep=None, engine='python')
             test_samples = df_test["sample"].astype(str).tolist()
        except:
             test_samples = []
    else:
        test_samples = []

    # (Skipping read regeneration check for simplicity, assume data step did it)

    print(f"[INFO] Found {len(test_samples)} test samples to process.")
    
    # 3. Align and Run HaplotypeCaller
    gvcfs = []
    for sample_name in test_samples:
        # Step A: Align to get BAM (always align to reference.fa)
            bam_path = runner.realign_sample(sample_name, reads_root, bowtie2_index)

        if not bam_path:
            continue

        # Step B: HaplotypeCaller
        gvcf = runner.run_haplotype_caller(bam_path, sample_name)
        if gvcf:
            gvcfs.append(gvcf)
            
    if not gvcfs:
        print("[ERROR] No GVCFs generated.")
        return

    # 4. Joint Genotyping
    final_joint_vcf = runner.vcf_dir / "gatk_test_cohort.vcf.gz"
    if final_joint_vcf.exists():
        print(f"[INFO] Final GATK VCF already exists: {final_joint_vcf}. Skipping Joint Genotyping.")
        gatk_vcf = final_joint_vcf
    else:
        gatk_vcf = runner.combine_and_genotype(gvcfs, out_vcf_name="gatk_test_cohort.vcf.gz")
    
    # 5. Merge with Reference VCF (Truth)
    truth_vcf = Path(cfg["project"]["simulate_vcfs"])
    ref_samples_csv = cfg["project"]["ref_samples"]
    ref_only_vcf = runner.vcf_dir / "ref_only_truth.vcf.gz"
    merged_vcf = runner.vcf_dir / "merged_gatk_truth.vcf.gz"
    
    # 5.1 Extract Reference Samples from Truth VCF
    if Path(ref_samples_csv).exists():
        ref_df = pd.read_csv(ref_samples_csv)
        ref_sample_ids = ref_df["sample"].astype(str).tolist()
        # Save sample list to file
        ref_list_file = runner.vcf_dir / "ref_samples.list"
        with open(ref_list_file, "w") as f:
            for s in ref_sample_ids:
                f.write(f"{s}\n")
        
        cmd_extract = [
            "bcftools", "view",
            "-S", str(ref_list_file),
            "-O", "z",
            "-o", str(ref_only_vcf),
            str(truth_vcf)
        ]
        subprocess.run(cmd_extract, check=True)
        subprocess.run(["bcftools", "index", str(ref_only_vcf)], check=True)
    else:
        print("[ERROR] ref_samples.csv not found.")
        return

    # 5.2 Merge GATK VCF (Test) + Ref VCF (Truth)
    print("[INFO] Merging GATK VCF (Test) with Truth VCF (Ref)...")
    cmd_merge = [
        "bcftools", "merge",
        "--force-samples", 
        "-O", "z",
        "-o", str(merged_vcf),
        str(gatk_vcf),
        str(ref_only_vcf)
    ]
    try:
        subprocess.run(cmd_merge, check=True)
        subprocess.run(["bcftools", "index", str(merged_vcf)], check=True)
        final_input_vcf = merged_vcf
        supervised_mode = True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Merge failed: {e}")
        return

    print(f"[INFO] Running ADMIXTURE (Supervised={supervised_mode}) on: {final_input_vcf}")


    # Infer K from labels file dynamically
    k = 3 
    ref_pops = ["AFR", "EUR", "ASIA", "YRI", "CEU", "NEA"] 
    
    labels_df = pd.read_csv(labels_path, sep=None, engine='python')
    
    if "population" in labels_df.columns:
        all_pops = sorted(labels_df["population"].unique().tolist())
        admix_pops = [p for p in all_pops if "ADMIX" in p.upper()]
        source_pops = [p for p in all_pops if p not in admix_pops]
        
        if admix_pops:
            k = len(source_pops)
            ref_pops = source_pops
            print(f"[AUTO] Detected Admixture mode. Target: {admix_pops}, Sources: {source_pops} (K={k})")
        else:
            k = len(all_pops)
            ref_pops = all_pops
            print(f"[AUTO] Detected Classification mode. Populations: {all_pops} (K={k})")
            
    print(f"[INFO] Using K={k}")

    # 6. Run ADMIXTURE
    run_admixture_from_vcf(
        vcf_path=str(final_input_vcf),
        sample_list_csv=labels_path,
        out_dir=str(out_dir / "admixture_result"),
        k=k,
        supervised=supervised_mode, 
        reference_pop_list=ref_pops,
        test_samples_path=test_samples_csv
    )

if __name__ == "__main__":
    run_admixture_pipeline()
