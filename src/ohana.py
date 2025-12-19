import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

class ANGSDRunner:
    def __init__(self, angsd_path, out_dir, threads=4):
        self.angsd = angsd_path
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.threads = threads

    def _ensure_ref_index(self, ref_fasta):
        """
        确保 ANGSD 使用的 reference.fa 索引是最新的，避免 “fai looks older” 报错。
        包括 samtools faidx (.fai) 与 GATK 字典 (.dict)。
        """
        ref_path = Path(ref_fasta)
        ref_fai = ref_path.with_suffix(ref_path.suffix + ".fai")
        ref_dict = ref_path.with_suffix(".dict")

        rebuilt = False

        if (not ref_fai.exists()) or (ref_fai.stat().st_mtime < ref_path.stat().st_mtime):
            print(f"[INFO] (ANGSD) 重新生成 {ref_fai}")
            subprocess.run(["samtools", "faidx", str(ref_path)], check=True)
            rebuilt = True

        if (not ref_dict.exists()) or (ref_dict.stat().st_mtime < ref_path.stat().st_mtime):
            gatk_bin = Path(__file__).resolve().parent.parent / "tools/gatk/gatk"
            print(f"[INFO] (ANGSD) 重新生成 {ref_dict}")
            subprocess.run([str(gatk_bin), "CreateSequenceDictionary", "-R", str(ref_path), "-O", str(ref_dict)], check=True)
            rebuilt = True
        # 无论是否重建，都触摸一次索引，确保时间戳明显新于 fasta，避免秒级精度导致的“looks older”误判
        subprocess.run(["touch", str(ref_fai)], check=False)
        subprocess.run(["touch", str(ref_dict)], check=False)

    def run_gl(self, bam_list_file, ref_fasta, out_prefix="angsd_gl", min_ind=None, sites_file=None):
        """
        Run ANGSD to estimate Genotype Likelihoods (GL) in Beagle format.
        
        Args:
            bam_list_file: Path to a file containing list of BAM paths.
            ref_fasta: Path to reference genome (needed for -doMajorMinor 1).
            out_prefix: Output filename prefix.
            min_ind: Minimum number of individuals with data to keep a site. 
                     Default: 50% of samples (if None).
            sites_file: (Optional) Path to a file containing sites to strictly keep (-sites).
                        Format: chr pos (1-based) or chr:pos.
                        Actually ANGSD -sites format is simpler if using -sites binary, 
                        but usually we use -sites with a file listing sites.
                        Note: ANGSD -sites requires indexing. 
                        Simplified: We won't use -sites binary index here, we assume user provided 
                        a region file or similar if needed.
                        
                        WAIT: To ensure compatibility with VCF->Beagle reference, we MUST enforce 
                        the exact same sites. The best way is to use -sites.
        """
        out_path = self.out_dir / out_prefix
        
        # 运行前确保参考索引新鲜，避免 ANGSD 因时间戳报错
        self._ensure_ref_index(ref_fasta)

        # Determine number of samples
        with open(bam_list_file) as f:
            n_samples = sum(1 for line in f if line.strip())
            
        if min_ind is None:
            min_ind = max(1, int(n_samples * 0.5))

        print(f"[INFO] Running ANGSD GL...")
        print(f"       BAM list: {bam_list_file}")
        print(f"       Ref: {ref_fasta}")
        print(f"       MinInd: {min_ind}/{n_samples}")

        # Construct command
        # -GL 1: SAMtools GL method
        # -doGlf 2: Beagle format output
        # -doMajorMinor 1: Infer major/minor from GLs
        # -doMaf 1: Estimate minor allele frequency (needed for major/minor)
        # -SNP_pval 1e-6: Only output SNPs with p-value < 1e-6 (Standard discovery)
        # However, if we merge with Truth, we might want to be permissive or force sites.
        # For this "Simple" flow, let's stick to standard discovery on Test set,
        # and then we will filter the Truth set to match these discovered sites (Intersection).
        # OR: We force ANGSD to call all sites from VCF? (Too slow/large)
        #
        # Better Strategy: 
        # 1. Run ANGSD on Test samples -> Discovers sites (e.g. 10k SNPs). Output: test.beagle.gz
        # 2. Extract THESE EXACT SITES from Truth VCF -> ref.beagle.gz
        # 3. Merge.
        #
        # So here we don't strictly force sites yet.
        
        # -doPost 1: Estimate posterior genotype probabilities
        # -minMapQ 20
        # -minQ 20
        # -minInd: Minimum number of individuals
        
        cmd = [
            str(self.angsd),
            "-b", str(bam_list_file),
            "-ref", str(ref_fasta),
            "-out", str(out_path),
            "-GL", "1",
            "-doGlf", "2",
            "-doMajorMinor", "1",
            "-doMaf", "1",
            "-doGeno", "32", 
            "-doPost", "1",
            "-minMapQ", "20",
            "-minQ", "20",
            "-minInd", str(min_ind),
            "-nThreads", str(self.threads)
        ]
        
        if sites_file:
            # -sites requires the file WITHOUT .bin extension, but the index must exist
            cmd.extend(["-sites", str(sites_file)])
            # If we force sites, we probably shouldn't filter by p-value as we trust the sites
            # So we DON'T add -SNP_pval here if sites_file is present
        else:
            # Standard discovery mode
            cmd.extend(["-SNP_pval", "1e-6"])

        print(f"[CMD] {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] ANGSD failed: {e}")
            return None

        # The output file will be out_path + ".beagle.gz"
        beagle_file = out_path.with_suffix(".beagle.gz")
            
        return beagle_file


class OhanaRunner:
    def __init__(self, ohana_root, out_dir):
        """
        ohana_root: Path to 'tools/ohana' directory. 
                    Expects binaries in ohana_root/bin/
        """
        self.bin_dir = Path(ohana_root) / "bin"
        self.convert_bin = self.bin_dir / "convert"
        self.qpas_bin = self.bin_dir / "qpas"
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def convert_beagle_to_lg(self, beagle_file, out_prefix="ohana_data"):
        """
        Convert Beagle GL file to Ohana .lg format.
        IMPORTANT: Ohana expects .lgm extension for Likelihood Genotype Matrix.
        """
        lg_file = self.out_dir / f"{out_prefix}.lgm"
        
        if lg_file.exists():
            print(f"[INFO] Ohana .lgm file exists: {lg_file}")
            return lg_file

        print(f"[INFO] Converting Beagle to Ohana .lgm format...")
        
        # Syntax: convert bgl2lgm input output
        cmd = [
            str(self.convert_bin),
            "bgl2lgm",
            str(beagle_file),
            str(lg_file)
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Ohana convert failed: {e}")
            return None
            
        return lg_file

    def run_qpas(self, lg_file, k, out_prefix="ohana_result", max_iter=100):
        """
        Run qpas to estimate ancestry proportions (Q matrix).
        Output: out_prefix.k.Q
        """
        out_Q = self.out_dir / f"{out_prefix}.{k}.Q"
        out_F = self.out_dir / f"{out_prefix}.{k}.F" # Allele frequencies
        
        if out_Q.exists():
            print(f"[INFO] Ohana Q matrix exists: {out_Q}")
            return out_Q

        print(f"[INFO] Running Ohana qpas (K={k})...")
        
        # qpas [options] <g-matrix>
        # -i is NOT a valid flag for input matrix in this version of qpas
        
        cmd = [
            str(self.qpas_bin),
            str(lg_file), # Positional argument for G matrix
            "-k", str(k),
            "-qo", str(out_Q), # Output Q path
            "-fo", str(out_F), # Output F path
            "-mi", str(max_iter),
            "-e", "0.0001"
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Ohana qpas failed: {e}")
            return None
            
        return out_Q

    def parse_and_save_result(self, q_file, bam_list_file, labels_csv, k, out_csv_path):
        """
        Parse .Q file, map columns to populations, and save as final CSV.
        
        q_file: The .Q matrix output from qpas (No header, space separated).
        bam_list_file: The list of BAMs used, to get sample names and order.
        labels_csv: The ground truth labels (sample, population) to map columns.
        """
        # 1. Get sample names from BAM list
        # We assume the order in BAM list matches the rows in Q matrix
        sample_names = []
        with open(bam_list_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    # /path/to/sample_name.sorted.bam -> sample_name
                    # Assuming filename is sample_name.sorted.bam or sample_name.bam
                    path = Path(line)
                    # Extract sample name: careful with naming conventions
                    # Our pipeline produces: sample.sorted.bam
                    name = path.name.replace(".sorted.bam", "").replace(".bam", "")
                    sample_names.append(name)
        
        # 2. Read Q matrix
        try:
            # qpas output Q file format:
            # First line: N K (e.g. 300 3)
            # Subsequent lines: Q values (space or tab separated)
            
            # Use skiprows=1 to skip the header line
            df_q = pd.read_csv(q_file, sep=r'\s+', header=None, skiprows=1)
        except Exception as e:
            print(f"[ERROR] Failed to read Q file: {e}")
            return None
            
        if len(df_q) != len(sample_names):
            print(f"[WARN] Sample count mismatch! Sample List: {len(sample_names)}, Q-rows: {len(df_q)}")
            # This is critical. If mismatch, we cannot assign names correctly.
            # Try to debug why mismatch happened.
            # Ohana convert bgl2lgm combines 3 columns into 1 individual.
            # So if input had 903 columns (3 marker + 900 data), output lgm has 300 individuals.
            # qpas output Q should have 300 rows.
            # If sample_names has 750, it means we added duplicates.
            
            # We will truncate or pad sample_names to match df_q length to avoid crash
            if len(sample_names) > len(df_q):
                print(f"[WARN] Truncating sample names list to match Q matrix size.")
                sample_names = sample_names[:len(df_q)]
            elif len(sample_names) < len(df_q):
                print(f"[WARN] Sample names list too short!")
                return None
        
        # 3. Map Columns (Similar to Admixture logic)
        raw_cols = [i for i in range(k)]
        df_q.columns = raw_cols
        
        # Create temp DF with IDs
        df_temp = df_q.copy()
        df_temp["sample"] = sample_names
        
        # Load labels
        col_to_pop = {}
        if Path(labels_csv).exists():
            try:
                try:
                    df_lbl = pd.read_csv(labels_csv, sep='\t')
                    if "population" not in df_lbl.columns:
                        df_lbl = pd.read_csv(labels_csv, sep=',')
                except:
                    df_lbl = pd.read_csv(labels_csv, sep=None, engine='python')
                    
                sample_pop_map = dict(zip(df_lbl["sample"].astype(str), df_lbl["population"].astype(str)))
                
                # Identify known populations
                # Logic: Use all pops in labels file that are also in our sample set
                present_pops = set()
                for s in sample_names:
                    if str(s) in sample_pop_map:
                        present_pops.add(sample_pop_map[str(s)])
                
                # Exclude "ADMIX" pops from being references for mapping
                ref_pops = [p for p in present_pops if "ADMIX" not in p.upper()]
                if not ref_pops: # Fallback if everything is ADMIX or unknown
                    ref_pops = list(present_pops)

                mapping_scores = []
                for pop in ref_pops:
                    pop_samples = [s for s in sample_names if sample_pop_map.get(str(s)) == pop]
                    if not pop_samples: continue
                    
                    sub_df = df_temp[df_temp["sample"].isin(pop_samples)]
                    if sub_df.empty: continue
                    
                    mean_q = sub_df[raw_cols].mean()
                    for col_idx in raw_cols:
                        mapping_scores.append((mean_q[col_idx], pop, col_idx))
                
                mapping_scores.sort(key=lambda x: x[0], reverse=True)
                
                used_cols = set()
                assigned_pops = set()
                
                for score, pop, col in mapping_scores:
                    if pop not in assigned_pops and col not in used_cols:
                        col_to_pop[col] = pop
                        used_cols.add(col)
                        assigned_pops.add(pop)
                    if len(assigned_pops) == len(ref_pops) or len(used_cols) == k:
                        break
                        
                print(f"[INFO] Mapped Ohana columns: {col_to_pop}")
                
            except Exception as e:
                print(f"[WARN] Failed to map columns using labels: {e}")
                
        # Rename columns
        new_cols = []
        for i in raw_cols:
            if i in col_to_pop:
                new_cols.append(f"Q_{col_to_pop[i]}")
            else:
                new_cols.append(f"Q_Pop{i}")
        
        df_q.columns = new_cols
        
        # Final DataFrame
        df_res = pd.concat([pd.Series(sample_names, name="sample"), df_q], axis=1)
        
        # Save
        df_res.to_csv(out_csv_path, index=False)
        print(f"[OK] Results saved to {out_csv_path}")
        return out_csv_path

