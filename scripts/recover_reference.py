import yaml
from pathlib import Path
import sys
import textwrap

# Add root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.read import SequencingRead

def recover_reference():
    print("[INFO] Recovering reference genome from random seed...")
    
    # Load config
    cfg = yaml.safe_load(Path("config/default.yaml").read_text())
    
    # Parameters matching run_read.py
    vcf_path = cfg["project"]["simulate_vcfs"]
    L = cfg["msprime"]["l"]
    read_root = cfg["project"]["reads_root"]
    # Default seed in SequencingRead is 42, assuming run_read.py used default
    seed = 42 
    
    print(f"[INFO] Initializing SequencingRead with L={L}, seed={seed}...")
    seq = SequencingRead(
        vcf_path=vcf_path,
        L=L,
        read_root=read_root,
        chrom="1",
        seed=seed
    )
    
    # The ref_array in SequencingRead is the reference sequence (Random Base + VCF REF)
    # This corresponds to the '1' chromosome in the VCF.
    ref_seq_array = seq.ref_array
    ref_seq = "".join(ref_seq_array)
    
    out_path = Path(cfg["project"].get("data_root", "/space/s1/qyx/data")) / "reference.fa"
    print(f"[INFO] Writing reference to {out_path}...")
    
    with open(out_path, "w") as f:
        f.write(">1\n")
        for chunk in textwrap.wrap(ref_seq, 100):
            f.write(chunk + "\n")
            
    print("[OK] Reference recovered.")
    
    # Also generate index
    import subprocess
    print("[INFO] Indexing reference...")
    subprocess.run(["samtools", "faidx", str(out_path)], check=True)
    
    # Generate dict
    dict_path = out_path.with_suffix(".dict")
    gatk_path = ROOT / "tools/gatk/gatk"
    if gatk_path.exists():
        print("[INFO] Generating sequence dictionary...")
        subprocess.run([
            str(gatk_path), "CreateSequenceDictionary",
            "-R", str(out_path),
            "-O", str(dict_path)
        ], check=True)
        
    print("[DONE] Reference ready for GATK.")

if __name__ == "__main__":
    recover_reference()

