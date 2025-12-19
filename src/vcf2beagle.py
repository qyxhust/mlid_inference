import gzip
import pysam
import numpy as np
from pathlib import Path
import pandas as pd

class VCF2Beagle:
    def __init__(self, vcf_path):
        self.vcf_path = Path(vcf_path)
        
    def convert(self, sample_list_path, out_prefix):
        """
        Convert VCF genotypes for specific samples to Beagle likelihood format.
        
        Beagle format (per line):
        marker allele1 allele2 P(AA) P(Aa) P(aa) ... for each sample
        
        Note: Ohana/ANGSD usually output P(MajorMajor) P(MajorMinor) P(MinorMinor)
        For Ref/Alt sites:
        0/0 -> 1.0 0.0 0.0
        0/1 -> 0.0 1.0 0.0
        1/1 -> 0.0 0.0 1.0
        """
        out_beagle = Path(out_prefix).with_suffix(".beagle.gz")
        
        # Load sample list
        if sample_list_path:
            df = pd.read_csv(sample_list_path)
            if 'sample' in df.columns:
                target_samples = df['sample'].astype(str).tolist()
            else:
                target_samples = []
        else:
            target_samples = []

        # Open VCF
        vcf = pysam.VariantFile(str(self.vcf_path))
        
        # Filter samples available in VCF
        vcf_samples = list(vcf.header.samples)
        valid_samples = [s for s in target_samples if s in vcf_samples]
        
        if not valid_samples:
            print("[ERROR] No valid samples found in VCF for conversion.")
            return None
            
        print(f"[INFO] Converting {len(valid_samples)} reference samples to Beagle format...")
        
        # Set subset
        vcf.subset_samples(valid_samples)
        
        with gzip.open(out_beagle, "wt") as f_out:
            # Header
            header = ["marker", "allele1", "allele2"]
            for s in valid_samples:
                header.extend([s, s, s]) # 3 columns per sample place holder names
            f_out.write("\t".join(header) + "\n")
            
            # Iterate variants
            count = 0
            for rec in vcf:
                # Basic info
                marker = f"{rec.chrom}_{rec.pos}"
                a1 = rec.ref
                a2 = rec.alts[0] if rec.alts else "N"
                
                line_parts = [marker, a1, a2]
                
                for s in valid_samples:
                    gt = rec.samples[s]["GT"]
                    # Default: Uniform likelihood (0.33, 0.33, 0.33) if missing
                    gls = ["0.333333", "0.333333", "0.333333"]
                    
                    if gt == (0, 0):
                        gls = ["1.000000", "0.000000", "0.000000"]
                    elif gt == (0, 1) or gt == (1, 0):
                        gls = ["0.000000", "1.000000", "0.000000"]
                    elif gt == (1, 1):
                        gls = ["0.000000", "0.000000", "1.000000"]
                        
                    line_parts.extend(gls)
                
                f_out.write("\t".join(line_parts) + "\n")
                count += 1
                
        print(f"[OK] Converted {count} sites to {out_beagle}")
        return out_beagle

