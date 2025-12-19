import yaml
import sys
import pandas as pd
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
sys.path.insert(0, str(ROOT))

from src.admixture import GATKRunner
from src.ohana import ANGSDRunner, OhanaRunner
from src.read import SequencingRead
from src.analysis import AncestryAnalysis

def run_ohana_pipeline(model_type="hard", supervised=True):
    # 1. Load Config
    cfg = yaml.safe_load(Path("config/default.yaml").read_text())
    
    # Check Reference
    if "ref_genome" in cfg["project"]:
        ref_fasta = Path(cfg["project"]["ref_genome"])
    else:
        # Fallback to old behavior
        data_root = Path(cfg["project"].get("data_root", "/space/s1/qyx/data"))
        ref_fasta = data_root / "reference.fa"
        if not ref_fasta.exists():
        ref_fasta = Path(cfg["project"]["ref_haps"])
    
    use_realigned = True # Always use realigned if possible, or generate it
    print(f"[INFO] Using reference genome: {ref_fasta}")

    reads_root = Path(cfg["project"]["reads_root"])
    align_root = Path(cfg["project"]["align_root"])
    
    # Output Directory
    # Determine mode for naming
    labels_path = cfg["project"]["simulate_label"]
    model_mode = model_type # Use passed argument as base
    
    # Naming convention: model_supervised_algo_depth_timestamp
    super_str = "supervised" if supervised else "unsupervised"
        
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    depth = cfg["reads"].get("depth", 1)
    algo = "angsd_ohana"
    
    # Format: model_algorithm_depth_timestamp
    run_name = f"{model_mode}_{algo}_{super_str}_{depth}x_{timestamp}"
    result_base = cfg["project"].get("result_base", "results")
    out_dir = Path(result_base) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Tools
    gatk_path = ROOT / "tools/gatk/gatk" # Used for re-alignment logic only
    angsd_path = ROOT / "tools/angsd/angsd"
    ohana_root = ROOT / "tools/ohana"
    
    print(f"[INFO] Ohana Pipeline Start (Supervised={supervised})")
    print(f"       Output: {out_dir}")
    
    # 2. Prepare BAMs (Use GATKRunner's logic to reuse/generate realigned BAMs)
    # We use GATKRunner solely for its alignment capabilities here
    align_runner = GATKRunner(gatk_path, ref_fasta, out_dir, align_root)
    # Ensure index exists for alignment
    align_runner.ensure_ref_index() 
    
    bowtie2_index = str(ref_fasta).rsplit(".", 1)[0]
    if use_realigned and not Path(bowtie2_index + ".1.bt2").exists():
        subprocess.run(["bowtie2-build", str(ref_fasta), bowtie2_index], check=True)

    # Collect Samples
    test_samples_csv = cfg["project"]["test_samples"]
    if Path(test_samples_csv).exists():
        try:
             df_test = pd.read_csv(test_samples_csv, sep=None, engine='python')
             test_samples = df_test["sample"].astype(str).tolist()
        except:
             test_samples = []
    else:
        test_samples = []

    # Regenerate Reads if missing (Copied logic)
    missing_reads = False
    if test_samples:
        for s in test_samples:
            r1 = reads_root / s / f"{s}_R1.fq"
            if not r1.exists():
                missing_reads = True
                break
    else:
        missing_reads = True

    if missing_reads:
        print("[INFO] Regenerating reads...")
        try:
            seq = SequencingRead(
                vcf_path=str(cfg["project"]["simulate_vcfs"]),
                L=cfg["msprime"]["l"],
                read_root=str(cfg["project"]["reads_root"]),
                chrom="1",
                test_table=str(test_samples_csv) if Path(test_samples_csv).exists() else None,
                seed=42
            )
            seq.write_test_haplotypes_per_sample()
            seq.run_wgsim_for_tests(depth=cfg["reads"]["depth"])
            if not test_samples:
                test_samples = list(seq.test_samples)
        except Exception as e:
            print(f"[ERROR] Failed to regenerate reads: {e}")
            return

    if not test_samples:
         test_samples = [d.name for d in reads_root.iterdir() if d.is_dir()]

    print(f"[INFO] Processing {len(test_samples)} samples...")
    
    bam_list = []
    for sample_name in test_samples:
        # Step A: Get BAM
        if use_realigned:
            # Check if GATK realigned bam exists first
            bam_path = align_runner.realigned_bam_dir / f"{sample_name}.sorted.bam"
            if not bam_path.exists():
                 bam_path = align_runner.realign_sample(sample_name, reads_root, bowtie2_index)
        else:
            bam_path = align_root / "bam" / f"{sample_name}.sorted.bam"
        
        if bam_path and bam_path.exists():
            bam_list.append(str(bam_path))
        else:
            print(f"[WARN] Failed to get BAM for {sample_name}")

    if not bam_list:
        print("[ERROR] No BAM files found.")
        return

    # Write BAM list to file
    bam_list_file = out_dir / "bam_list.txt"
    with open(bam_list_file, "w") as f:
        for b in bam_list:
            f.write(b + "\n")
            
    # 3. Run ANGSD (GL Estimation) - Test Samples Only
    angsd_runner = ANGSDRunner(angsd_path, out_dir)
    truth_vcf = Path(cfg["project"]["simulate_vcfs"])
    
    if supervised:
        # Strategy: Force ANGSD to use ALL sites from Truth VCF to ensure high SNP count and alignment
        print("[INFO] Extracting sites from Truth VCF for ANGSD (Supervised)...")
        sites_file = out_dir / "sites.txt"
        
        import gzip
        with gzip.open(truth_vcf, 'rt') as f_in, open(sites_file, 'w') as f_out:
            for line in f_in:
                if not line.startswith('#'):
                    parts = line.split('\t', 2)
                    # ANGSD sites file format: chrom position (1-based)
                    f_out.write(f"{parts[0]}\t{parts[1]}\n")
                    
        # Index sites file for ANGSD
        print("[INFO] Indexing sites file for ANGSD...")
        # Clean up old index files if any
        if (out_dir / "sites.txt.bin").exists():
            (out_dir / "sites.txt.bin").unlink()
        if (out_dir / "sites.txt.idx").exists():
            (out_dir / "sites.txt.idx").unlink()
            
        import time
        time.sleep(1) # Ensure timestamp diff
        
        subprocess.run([str(angsd_path), "sites", "index", str(sites_file)], check=True)
        
        # Ensure index is newer than sites file
        sites_bin = out_dir / "sites.txt.bin"
        if sites_bin.exists():
            subprocess.run(["touch", str(sites_bin)], check=True)
            
        # min_ind=0 to force output for all sites
        test_beagle = angsd_runner.run_gl(bam_list_file, ref_fasta, min_ind=0, sites_file=sites_file)
    else:
        # Unsupervised: Use same sites but do not merge ref
        # If we use sites from truth vcf, we get comparable sites but no ref samples.
        print("[INFO] Extracting sites from Truth VCF for ANGSD (Unsupervised)...")
        sites_file = out_dir / "sites.txt"
        import gzip
        with gzip.open(truth_vcf, 'rt') as f_in, open(sites_file, 'w') as f_out:
            for line in f_in:
                if not line.startswith('#'):
                    parts = line.split('\t', 2)
                    f_out.write(f"{parts[0]}\t{parts[1]}\n")
                    
        print("[INFO] Indexing sites file for ANGSD...")
        if (out_dir / "sites.txt.bin").exists():
            (out_dir / "sites.txt.bin").unlink()
        if (out_dir / "sites.txt.idx").exists():
            (out_dir / "sites.txt.idx").unlink()
            
        import time
        time.sleep(1)
        
        subprocess.run([str(angsd_path), "sites", "index", str(sites_file)], check=True)
        
        # Ensure index is newer
        sites_bin = out_dir / "sites.txt.bin"
        if sites_bin.exists():
            subprocess.run(["touch", str(sites_bin)], check=True)
            
        test_beagle = angsd_runner.run_gl(bam_list_file, ref_fasta, min_ind=0, sites_file=sites_file)
    
    if not test_beagle or not test_beagle.exists():
        print("[ERROR] ANGSD failed to produce beagle file.")
        return

    # 4. Prepare Reference GL (Supervised Only)
    unified_sample_list = []
    final_beagle = test_beagle
    
    if supervised:
        print("[INFO] Preparing Reference GLs from Truth VCF...")
        ref_samples_csv = cfg["project"]["ref_samples"]
        
        from src.vcf2beagle import VCF2Beagle
        
        # 4.1 Generate Ref Beagle
        v2b = VCF2Beagle(str(truth_vcf))
        ref_beagle = v2b.convert(ref_samples_csv, str(out_dir / "ref_data"))
        
        # 4.2 Merge Test and Ref Beagle
        merged_beagle = out_dir / "merged_cohort.beagle" 
        print(f"[INFO] Merging Test and Ref Beagle files to {merged_beagle}...")
        
        try:
            df_test = pd.read_csv(test_beagle, sep='\t', compression='gzip')
            df_ref = pd.read_csv(ref_beagle, sep='\t', compression='gzip')
            
            cols_to_use = ['marker'] + [c for c in df_ref.columns if c not in ['marker', 'allele1', 'allele2']]
            df_ref_sub = df_ref[cols_to_use]
            
            df_merged = pd.merge(df_test, df_ref_sub, on='marker', how='inner')
            
            print(f"[INFO] Merged sites: {len(df_merged)}")
            if len(df_merged) == 0:
                print("[ERROR] No overlapping sites found between Test and Ref!")
                return

            df_merged.to_csv(merged_beagle, sep='\t', index=False)
            final_beagle = merged_beagle
            
            # Construct Unified Sample List
            # 1. Test Samples
            with open(bam_list_file) as f:
                for line in f:
                    if line.strip():
                        name = Path(line.strip()).name.replace(".sorted.bam", "").replace(".bam", "")
                        unified_sample_list.append(name)
                        
            # 2. Ref Samples
            ref_cols = [c for c in df_ref.columns if c not in ['marker', 'allele1', 'allele2']]
            seen = set()
            for c in ref_cols:
                if c not in seen:
                    unified_sample_list.append(c)
                    seen.add(c)
            
        except Exception as e:
            print(f"[ERROR] Merge failed: {e}")
            return
            
    else:
        # Unsupervised: Just use Test Beagle
        print(f"[INFO] Decompressing Beagle file for Ohana...")
        uncompressed_beagle = out_dir / "test_cohort.beagle"
        with gzip.open(test_beagle, 'rt') as f_in, open(uncompressed_beagle, 'w') as f_out:
            for line in f_in:
                f_out.write(line)
        final_beagle = uncompressed_beagle
        
        # Sample list is just test samples
        with open(bam_list_file) as f:
            for line in f:
                if line.strip():
                    name = Path(line.strip()).name.replace(".sorted.bam", "").replace(".bam", "")
                    unified_sample_list.append(name)

    print(f"[INFO] Unified sample list size: {len(unified_sample_list)}")
    
    # Write this list to a file for OhanaRunner to use
    unified_list_file = out_dir / "unified_sample_list.txt"
    with open(unified_list_file, "w") as f:
        # We write dummy paths just to satisfy the parser which expects paths
        for s in unified_sample_list:
            f.write(f"/dummy/path/{s}.bam\n")
            
    # Run Ohana (Inference)
    ohana_runner = OhanaRunner(ohana_root, out_dir)
    
    # Ensure LD_LIBRARY_PATH includes conda lib for ohana binaries
    import os
    conda_lib = Path(sys.executable).parent.parent / "lib"
    env_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if str(conda_lib) not in env_ld:
        os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:{env_ld}"
    
    # Convert
    lg_file = ohana_runner.convert_beagle_to_lg(final_beagle)
    if not lg_file: return
    
    # Infer K from labels
    k = 3
    if Path(labels_path).exists():
        try:
            df_lbl = pd.read_csv(labels_path, sep=None, engine='python')
            if "population" in df_lbl.columns:
                all_pops = sorted(df_lbl["population"].unique().tolist())
                admix_pops = [p for p in all_pops if "ADMIX" in p.upper()]
                source_pops = [p for p in all_pops if p not in admix_pops]
                if admix_pops:
                    k = len(source_pops)
                else:
                    k = len(all_pops)
        except:
            pass
    print(f"[INFO] Using K={k}")

    # Run qpas
    q_file = ohana_runner.run_qpas(lg_file, k)
    if not q_file: return
    
    # Parse Result
    final_csv = out_dir / "ohana_final_result.csv"
    ohana_runner.parse_and_save_result(q_file, unified_list_file, labels_path, k, final_csv)
    
    # 6. Plotting

    if final_csv.exists():
        ana = AncestryAnalysis(str(final_csv), labels_path, str(out_dir), test_samples_path=test_samples_csv)
        
        if model_mode == "admix":
            print("[ANALYSIS] Plotting structure barplot...")
            ana.plot_admixture_barplot(top_n=200, only_test=True)
        else:
            print("[ANALYSIS] Plotting confusion matrix...")
            ana.plot_confusion_matrix()

if __name__ == "__main__":
    run_ohana_pipeline()