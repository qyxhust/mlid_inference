import subprocess
from pathlib import Path
import pandas as pd
import sys

# 假设 ROOT/scripts 在 path 中，或者我们使用相对导入如果 analysis 移到了 src
# 目前 analysis 在 scripts/analysis.py
# 为了避免循环引用或路径问题，我们在函数内部 import 或者假设环境已配置好
# 这里我们假设 pipeline/script 会设置好 sys.path

class GATKRunner:
    def __init__(self, gatk_path, ref_fasta, out_dir, align_root, threads=4):
        self.gatk = gatk_path
        self.ref = Path(ref_fasta)
        self.out_dir = Path(out_dir) # Only for final results
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.threads = threads
        
        # Centralized data storage for intermediate files
        self.align_root = Path(align_root)
        self.realigned_bam_dir = self.align_root / "realigned_bam"
        self.realigned_bam_dir.mkdir(parents=True, exist_ok=True)
        
        self.gvcf_dir = self.align_root / "gvcfs"
        self.gvcf_dir.mkdir(parents=True, exist_ok=True)
        
        self.vcf_dir = self.align_root / "vcfs"
        self.vcf_dir.mkdir(parents=True, exist_ok=True)

    def ensure_ref_index(self):
        """
        GATK requires .fai and .dict files for the reference.
        """
        # 1. fasta index (.fai)
        if not self.ref.with_suffix(".fa.fai").exists() and not self.ref.with_suffix(".fai").exists():
            print("[INFO] Generating reference index (.fai)...")
            subprocess.run(["samtools", "faidx", str(self.ref)], check=True)
            
        # 2. sequence dictionary (.dict)
        # .dict usually replaces the extension, e.g., ref.fa -> ref.dict
        dict_path = self.ref.with_suffix(".dict")
        if not dict_path.exists():
            print("[INFO] Generating reference dictionary (.dict)...")
            cmd = [
                str(self.gatk), "CreateSequenceDictionary",
                "-R", str(self.ref),
                "-O", str(dict_path)
            ]
            subprocess.run(cmd, check=True)

    def realign_sample(self, sample_name, reads_root, bowtie2_index):
        """
        Align sample reads to the reference.fa to generate a clean BAM for GATK.
        Saves BAM to align_root/realigned_bam/
        """
        # Reads location: data/reads/<sample>/<sample>_R1.fq
        r1 = reads_root / sample_name / f"{sample_name}_R1.fq"
        r2 = reads_root / sample_name / f"{sample_name}_R2.fq"
        
        if not r1.exists() or not r2.exists():
             print(f"[ERROR] Reads not found for {sample_name}: {r1}")
             return None

        out_bam = self.realigned_bam_dir / f"{sample_name}.sorted.bam"
        
        if out_bam.exists():
            return out_bam
            
        print(f"[INFO] Aligning {sample_name} to reference...")
        
        # Bowtie2 alignment
        # Use a pipe to sort directly
        cmd_bt = [
            "bowtie2",
            "--all", "--very-sensitive",
            "-p", str(self.threads),
            "-x", bowtie2_index,
            "-1", str(r1),
            "-2", str(r2),
            "--rg-id", "1",
            "--rg", f"SM:{sample_name}",
            "--rg", "PL:illumina"
        ]
        
        cmd_sort = ["samtools", "sort", "-o", str(out_bam)]
        
        p1 = subprocess.Popen(cmd_bt, stdout=subprocess.PIPE)
        p2 = subprocess.Popen(cmd_sort, stdin=p1.stdout)
        p1.stdout.close()
        p2.communicate()
        
        if p2.returncode != 0:
            print(f"[ERROR] Alignment failed for {sample_name}")
            return None
            
        subprocess.run(["samtools", "index", str(out_bam)], check=True)
        return out_bam

    def run_haplotype_caller(self, bam_path, sample_name):
        """
        Run HaplotypeCaller in GVCF mode for a single sample.
        Saves GVCF to align_root/gvcfs/
        """
        out_gvcf = self.gvcf_dir / f"{sample_name}.g.vcf.gz"
        
        if out_gvcf.exists():
            print(f"[INFO] GVCF already exists for {sample_name}, skipping.")
            return out_gvcf

        print(f"[INFO] Running HaplotypeCaller for {sample_name}...")
        cmd = [
            str(self.gatk), "HaplotypeCaller",
            "-R", str(self.ref),
            "-I", str(bam_path),
            "-O", str(out_gvcf),
            "-ERC", "GVCF",
            "--native-pair-hmm-threads", str(self.threads),
            # Fix for Bowtie2 MAPQ=255 (which GATK treats as unavailable)
            "--disable-read-filter", "MappingQualityAvailableReadFilter"
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] HaplotypeCaller failed for {sample_name}: {e}")
            return None
            
        return out_gvcf

    def combine_and_genotype(self, gvcf_list, out_vcf_name="final_joint.vcf.gz"):
        """
        CombineGVCFs -> GenotypeGVCFs
        Saves VCFs to align_root/vcfs/
        """
        combined_gvcf = self.vcf_dir / "cohort.g.vcf.gz"
        final_vcf = self.vcf_dir / out_vcf_name
        
        # 1. CombineGVCFs
        if not combined_gvcf.exists():
            print("[INFO] Running CombineGVCFs...")
            
            args_file = self.vcf_dir / "gvcfs.list"
            with open(args_file, "w") as f:
                for p in gvcf_list:
                    f.write(str(p) + "\n")
            
            cmd = [
                str(self.gatk), "CombineGVCFs",
                "-R", str(self.ref),
                "-V", str(args_file),
                "-O", str(combined_gvcf)
            ]
            subprocess.run(cmd, check=True)
        else:
            print(f"[INFO] Combined GVCF exists: {combined_gvcf}")

        # 2. GenotypeGVCFs
        if not final_vcf.exists():
            print("[INFO] Running GenotypeGVCFs...")
            cmd = [
                str(self.gatk), "GenotypeGVCFs",
                "-R", str(self.ref),
                "-V", str(combined_gvcf),
                "-O", str(final_vcf)
            ]
            subprocess.run(cmd, check=True)
        else:
             print(f"[INFO] Final VCF exists: {final_vcf}")
             
        return final_vcf


def run_admixture_from_vcf(
    vcf_path: str,
    sample_list_csv: str,
    out_dir: str,
    k: int = 3,
    threads: int = 4,
    plink_path: str = "plink2",
    admixture_path: str = None, # Should be provided
    supervised: bool = False,
    reference_pop_list: list = None,
    test_samples_path: str = None
):
    """
    1. Convert VCF -> PLINK (bed/bim/fam)
    2. Run ADMIXTURE
    3. Plot results (calls scripts.analysis.AncestryAnalysis)
    """
    
    # Delayed import to avoid circular dependency if analysis is in scripts
    # and scripts imports src.
    from src.analysis import AncestryAnalysis

    if admixture_path is None:
         # Fallback default
         current_dir = Path(__file__).resolve().parent
         admixture_path = str(current_dir.parent / "tools/admixture/admixture")

    vcf_path = Path(vcf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = out_dir / "admixture_input"
    
    print(f"[INFO] Converting VCF to PLINK format...")
    print(f"       VCF: {vcf_path}")
    print(f"       Out: {prefix}")

    # 1. VCF -> PLINK
    cmd_plink = [
        plink_path,
        "--vcf", str(vcf_path),
        "--make-bed",
        "--out", str(prefix),
        "--allow-extra-chr",
        "--max-alleles", "2", 
        "--geno", "0.1",      
        "--mind", "0.99",   # Relaxed missingness filter for low-coverage data
        "--threads", str(threads)
    ]
    
    try:
        subprocess.run(cmd_plink, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] PLINK conversion failed: {e}")
        return

    bed_file = prefix.with_suffix(".bed")
    if not bed_file.exists():
        print(f"[ERROR] PLINK output not found: {bed_file}")
        return

    # 2. Run ADMIXTURE
    print(f"[INFO] Running ADMIXTURE (K={k}, Supervised={supervised})...")
    
    cmd_admix = [
        admixture_path,
        "--cv",
        f"{prefix.name}.bed", # Use just the filename
        str(k),
        "-j" + str(threads)
    ]
    
    # === Handle Supervised Mode ===
    if supervised:
        fam_file = prefix.with_suffix(".fam")
        pop_file = prefix.with_suffix(".pop")
        
        df_fam = pd.read_csv(fam_file, sep=r'\s+', header=None, names=["FID", "IID", "F", "M", "S", "P"])
        df_labels = pd.read_csv(sample_list_csv, sep=None, engine='python')
        sample_pop_map = dict(zip(df_labels["sample"].astype(str), df_labels["population"].astype(str)))
        
        pop_values = []
        
        test_sample_ids = set()
        if test_samples_path and Path(test_samples_path).exists():
            try:
                df_test = pd.read_csv(test_samples_path, sep=None, engine='python')
                if "sample" in df_test.columns:
                    test_sample_ids = set(df_test["sample"].astype(str))
            except Exception as e:
                print(f"[WARN] Failed to read test samples for filtering: {e}")

        ref_pops_set = set(reference_pop_list) if reference_pop_list else set()
        
        for iid in df_fam["IID"]:
            iid = str(iid)
            if iid in test_sample_ids:
                pop_values.append("-")
                continue

            if iid in sample_pop_map:
                real_pop = sample_pop_map[iid]
                is_ref = False
                if reference_pop_list:
                    if real_pop in ref_pops_set:
                        is_ref = True
                else:
                    if "ADMIX" not in real_pop.upper():
                         is_ref = True
                
                if is_ref:
                    pop_values.append(real_pop)
                else:
                    pop_values.append("-")
            else:
                pop_values.append("-")
        
        with open(pop_file, "w") as f:
            for p in pop_values:
                f.write(f"{p}\n")
                
        cmd_admix.append("--supervised")

    try:
        subprocess.run(cmd_admix, cwd=out_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ADMIXTURE failed: {e}")
        return

    # 3. Parse Output (.Q file)
    q_file = out_dir / f"{prefix.name}.{k}.Q"
    fam_file = prefix.with_suffix(".fam")
    
    if not q_file.exists():
        print(f"[ERROR] .Q file not found: {q_file}")
        return
    
    print(f"[INFO] Parsing results: {q_file}")
    
    df_fam = pd.read_csv(fam_file, sep=r'\s+', header=None, names=["FID", "IID", "F", "M", "S", "P"])
    samples = df_fam["IID"].tolist()
    
    df_q = pd.read_csv(q_file, sep=r'\s+', header=None)
    
    # === Column Mapping Logic ===
    raw_cols = [i for i in range(k)]
    df_q.columns = raw_cols
    df_temp = pd.concat([df_fam[["IID"]], df_q], axis=1)
    
    if Path(sample_list_csv).exists():
        try:
            df_labels = pd.read_csv(sample_list_csv, sep=None, engine='python')
        except:
            pass

        sample_pop_map = dict(zip(df_labels["sample"].astype(str), df_labels["population"].astype(str)))
        pop_to_col = {}
        known_pops = sorted(list(set(reference_pop_list) & set(df_labels["population"].unique())))
        mapping_scores = []
        
        for pop in known_pops:
            pop_samples = [s for s in samples if sample_pop_map.get(str(s)) == pop]
            if not pop_samples: continue
            sub_df = df_temp[df_temp["IID"].isin(pop_samples)]
            if sub_df.empty: continue
            mean_q = sub_df[raw_cols].mean()
            for col_idx in raw_cols:
                mapping_scores.append((mean_q[col_idx], pop, col_idx))

        mapping_scores.sort(key=lambda x: x[0], reverse=True)
        used_cols = set()
        assigned_pops = set()
        for score, pop, col in mapping_scores:
            if pop not in assigned_pops and col not in used_cols:
                pop_to_col[pop] = col
                used_cols.add(col)
                assigned_pops.add(pop)
            if len(assigned_pops) == len(known_pops) or len(used_cols) == len(raw_cols): break

        col_to_pop = {v: k for k, v in pop_to_col.items()}
        new_columns = []
        for i in raw_cols:
            if i in col_to_pop:
                new_columns.append(f"Q_{col_to_pop[i]}")
            else:
                new_columns.append(f"Q_Pop{i}")
        df_q.columns = new_columns
        print(f"[INFO] Mapped ADMIXTURE columns: {col_to_pop}")
    else:
        df_q.columns = [f"Q_Pop{i+1}" for i in range(k)]
    
    df_res = pd.concat([df_fam[["IID"]], df_q], axis=1)
    df_res = df_res.rename(columns={"IID": "sample"})
    
    final_csv = out_dir / "admixture_final_result.csv"
    df_res.to_csv(final_csv, index=False)
    print(f"[OK] Saved parsed results to {final_csv}")

    # 4. Plotting
    if Path(sample_list_csv).exists():
        print(f"[INFO] Plotting using labels from: {sample_list_csv}")
        ana = AncestryAnalysis(str(final_csv), sample_list_csv, str(out_dir), test_samples_path=test_samples_path)
        
        is_admix_mode = False
        try:
             df_lbl = pd.read_csv(sample_list_csv, sep=None, engine='python')
             if "population" in df_lbl.columns:
                 if any("ADMIX" in str(p).upper() for p in df_lbl["population"].unique()):
                     is_admix_mode = True
        except:
            pass

        try:
            if is_admix_mode:
                print("[ANALYSIS] Detected Admixture mode. Plotting structure barplot...")
                ana.plot_admixture_barplot(top_n=200, only_test=True)
            else:
                print("[ANALYSIS] Detected Classification (Hard) mode. Plotting confusion matrix...")
                ana.plot_confusion_matrix()
        except Exception as e:
            print(f"[WARN] Plotting failed: {e}")

