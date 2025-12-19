import subprocess
from pathlib import Path

import pandas as pd
import pysam


class EMAlignment:
    def __init__(
        self,
        ref_fasta_path: str,
        reads_root: str,
        out_root: str,
        index_prefix: str | None = None,
    ):
        self.ref_fasta_path = Path(ref_fasta_path)
        self.reads_root = Path(reads_root)

        self.out_root = Path(out_root)
        self.align_dir = self.out_root / "bam"
        self.align_dir.mkdir(parents=True, exist_ok=True)

        if index_prefix is None:
            self.index_prefix = str(self.ref_fasta_path).rsplit(".", 1)[0] + "_index"
        else:
            self.index_prefix = str(index_prefix)

    def build_bowtie2_index(self, bowtie2_build: str = "bowtie2-build"):
        bt2_file = self.index_prefix + ".1.bt2"
        if Path(bt2_file).exists():
            print("[INFO] bowtie2 index 已存在，跳过构建：", self.index_prefix)
            return

        cmd = [bowtie2_build, str(self.ref_fasta_path), self.index_prefix]
        print("[INFO] Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print("[OK] bowtie2 index built:", self.index_prefix)

    def write_hap_index_table(self, out_csv: str | None = None):
        if out_csv is None:
            out_csv = self.out_root / "ref_haps_index.csv"
        else:
            out_csv = Path(out_csv)

        fasta = pysam.FastaFile(str(self.ref_fasta_path))
        refs = list(fasta.references)
        rows = [{"hap_index": i, "hap_id": name} for i, name in enumerate(refs)]
        fasta.close()

        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        print("[OK] hap index table ->", out_csv)
        return df

    def ensure_bam(
        self,
        sample_id: str,
        bowtie2_path: str = "bowtie2",
        samtools_path: str = "samtools",
        threads: int = 4,
        delete_reads_after: bool = True,
    ) -> Path:
        """
        确保某个 sample 的 BAM 存在：
          - 如果 bam 已存在 -> 直接返回
          - 否则从 fastq 开始做对齐；
            成功后可以选择删除 fastq
        """
        bam_path = self.align_dir / f"{sample_id}.sorted.bam"
        if bam_path.exists():
            print(f"[INFO] BAM 已存在，跳过对齐: {bam_path}")
            return bam_path

        r1 = self.reads_root / sample_id / f"{sample_id}_R1.fq"
        r2 = self.reads_root / sample_id / f"{sample_id}_R2.fq"
        if not r1.exists() or not r2.exists():
            raise FileNotFoundError(f"找不到 fastq：{r1} 或 {r2}")

        sam_path = self.align_dir / f"{sample_id}.sam"

        print(f"[INFO] 开始对齐 sample={sample_id}")
        cmd_bt = [
            bowtie2_path,
            "--all",
            "--very-sensitive",
            "-p",
            str(threads),
            "-x",
            self.index_prefix,
            "-1",
            str(r1),
            "-2",
            str(r2),
            "-S",
            str(sam_path),
        ]
        print("[INFO] bowtie2 cmd:\n       " + " ".join(cmd_bt))
        subprocess.run(cmd_bt, check=True)

        cmd_view = [samtools_path, "view", "-bS", str(sam_path)]
        # 修改为按名称排序 (-n)，这对 TaxIdent/MLID 处理多重比对至关重要
        cmd_sort = [samtools_path, "sort", "-n", "-o", str(bam_path)]
        print("[INFO] samtools view|sort -n (by name) for sample:", sample_id)
        p1 = subprocess.Popen(cmd_view, stdout=subprocess.PIPE)
        p2 = subprocess.Popen(cmd_sort, stdin=p1.stdout)
        p1.stdout.close()
        p2.communicate()
        if p2.returncode != 0:
            raise RuntimeError(f"samtools sort 失败 for sample {sample_id}")

        # 删除 SAM
        try:
            sam_path.unlink()
        except OSError:
            pass

        print("[OK] BAM written:", bam_path)

        # ✓ 对齐成功后删除 R1/R2
        if delete_reads_after:
            for fq in [r1, r2]:
                try:
                    fq.unlink()
                    print(f"[INFO] 删除 fastq: {fq}")
                except OSError:
                    pass

        return bam_path
