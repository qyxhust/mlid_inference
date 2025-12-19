import subprocess
from pathlib import Path
import pandas as pd


class TaxidentRunner:
    def __init__(self, taxident_path: str, out_root: str):
        self.taxident_path = taxident_path
        self.out_root = Path(out_root)
        self.tax_dir = self.out_root / "taxident"
        self.tax_dir.mkdir(parents=True, exist_ok=True)

    def taxident_output_path(self, sample_id: str) -> Path:
        return self.tax_dir / f"{sample_id}.dense.txt"

    def ensure_taxident(
        self,
        sample_id: str,
        bam_path: Path,
        delete_bam_after: bool = True,
        threads: int = 1,
    ) -> Path:
        """
        确保 sample 的 TaxIdent 输出存在：
          - 如果 dense 已存在 -> 直接返回
          - 否则从 BAM 调用 TaxIdent；
            成功后可删除 BAM
        """
        out_path = self.taxident_output_path(sample_id)
        if out_path.exists():
            print(f"[INFO] TaxIdent 输出已存在，跳过: {out_path}")
            return out_path

        cmd = [
            self.taxident_path,
            "-i",
            str(bam_path),
            "-o",
            str(out_path),
            "--format",
            "dense",
            "--dense",
            "--no-damage",
            "-t",
            str(threads),
        ]
        print("[INFO] TaxIdent for sample:", sample_id)
        print("       " + " ".join(cmd))
        subprocess.run(cmd, check=True)

        print("[OK] TaxIdent output:", out_path)

        # ✓ TaxIdent 成功后删除 BAM
        if delete_bam_after:
            try:
                bam_path.unlink()
                print(f"[INFO] 删除 BAM: {bam_path}")
            except OSError:
                pass

        return out_path


class EMRunner:
    def __init__(
        self,
        hap_index_source: str | pd.DataFrame,
        out_root: str,
        result_csv: str,
        mlid_path: str = None,
        final_result_csv: str = None,
    ):
        if isinstance(hap_index_source, pd.DataFrame):
            self.hap_index_df = hap_index_source
        else:
            self.hap_index_df = pd.read_csv(hap_index_source)
            
        self.out_root = Path(out_root)
        self.result_csv = Path(result_csv)
        self.final_result_csv = Path(final_result_csv) if final_result_csv else None

        if mlid_path is None:
            # 尝试默认路径: 相对于当前文件路径的 ../tools/MLID/mlid
            current_dir = Path(__file__).resolve().parent
            mlid_path = current_dir.parent / "tools/MLID/mlid"

        self.mlid_path = Path(mlid_path)
        if not self.mlid_path.exists():
            print(f"[WARNING] mlid executable not found at: {self.mlid_path}")

    def _sample_in_result(self, sample_id: str) -> bool:
        # 优先检查 final_result.csv (Population level)
        # 1. 优先使用显式传入的 final_result_csv
        # 2. 如果没有，尝试推断
        final_res_path = None
        if self.final_result_csv:
             final_res_path = self.final_result_csv
        else:
             final_res_path = Path(str(self.result_csv).replace("em_results.csv", "final_result.csv"))
        
        found_in_final = False
        if final_res_path and final_res_path.exists():
             try:
                 df = pd.read_csv(final_res_path)
                 if "sample" in df.columns:
                     # 调试信息: 打印前几个样本，确认读取正确
                     # print(f"[DEBUG] Checking {final_res_path}, samples found: {df['sample'].head().tolist()}")
                     if str(sample_id) in df["sample"].astype(str).tolist():
                         found_in_final = True
                     else:
                         # 只有当找不到时才打印 DEBUG，避免刷屏
                         # print(f"[DEBUG] Sample {sample_id} NOT found in {final_res_path}")
                         pass
             except Exception as e:
                 print(f"[WARNING] Failed to read {final_res_path}: {e}")
        else:
             # print(f"[DEBUG] Final result file not found: {final_res_path}")
             pass

        if found_in_final:
            return True

        # 回退到检查 em_results
        if not self.result_csv.exists():
            return False
        
        try:
            df = pd.read_csv(self.result_csv)
            if "sample" not in df.columns:
                return False
            return str(sample_id) in df["sample"].astype(str).tolist()
        except Exception as e:
             print(f"[WARNING] Failed to read {self.result_csv}: {e}")
             return False

    def _parse_mlid_output(self, mlid_out_path: Path):
        """
        解析 MLID 输出文件，提取 loglik 和 proportions
        MLID 输出格式示例:
        Model=BEfull
        LogLikelihood=-12345.678901
        ...
        genome	proportion	error_rate	damage_rate
        0	0.800000	0.000099	not_applicable
        1	0.100000	0.000099	not_applicable
        ...
        """
        loglik = None
        p_pop = {}
        
        if not mlid_out_path.exists():
            raise FileNotFoundError(f"MLID output not found: {mlid_out_path}")

        with open(mlid_out_path, "r") as f:
            lines = f.readlines()

        # 1. 解析 Metadata
        data_start_idx = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            if line.startswith("LogLikelihood="):
                try:
                    loglik = float(line.split("=")[1])
                except ValueError:
                    loglik = float("nan")
            if line.startswith("genome"):  # Header line of data section
                data_start_idx = i
                break
        
        if loglik is None:
             print(f"[WARNING] Could not find LogLikelihood in {mlid_out_path}")
             loglik = 0.0

        # 2. 解析 Data Section
        # 使用 pandas 读取剩余部分
        try:
            # 从 data_start_idx 开始读取，使用 tab 分隔
            # 注意: read_csv 可以接受 skip_rows
            df_res = pd.read_csv(mlid_out_path, sep="\t", skiprows=data_start_idx)
            
            # 构建 p_pop 字典: {hap_id: proportion}
            # 需要将 genome index (0, 1, 2...) 映射回 hap_id (REF1, REF2...)
            # hap_index_df columns: hap_index, hap_id
            
            # 创建 mapping: hap_index (str) -> hap_id
            index_map = dict(zip(
                self.hap_index_df["hap_index"].astype(str), 
                self.hap_index_df["hap_id"]
            ))
            
            for _, row in df_res.iterrows():
                g_idx = str(row["genome"]) # TaxIdent 输出的可能是 index (0, 1) 也可能是 (G1, G2)
                
                # 修复: TaxIdent Dense 模式输出 Header 为 G1, G2... 对应 Index 0, 1...
                # 如果遇到 G 开头的数字，将其转换为对应的 0-based index
                if g_idx.startswith("G") and g_idx[1:].isdigit():
                    g_idx = str(int(g_idx[1:]) - 1)

                prop = row["proportion"]
                
                # 如果 g_idx 在 mapping 中，用 mapping 后的名字，否则保留原名
                if g_idx in index_map:
                    real_name = index_map[g_idx]
                    p_pop[real_name] = prop
                else:
                    # 可能是 'T1:genus' 这种 TaxIdent 生成的名字，或者 mapping 缺失
                    p_pop[g_idx] = prop
                    
        except Exception as e:
            print(f"[ERROR] Failed to parse MLID data section: {e}")

        return loglik, p_pop

    def run_em_if_needed(
        self,
        sample_id: str,
        taxident_out: Path,
        delete_taxident_after: bool = True,
    ):
        """
        逻辑：
          - 如果 em_results.csv 中已有此 sample -> 直接跳过
          - 否则：
              * 用 taxident_out + hap_index_df 跑 EM
              * 把结果追加写入 em_results.csv
              * 成功后删除 taxident_out 文件
        """
        if self._sample_in_result(sample_id):
            print(f"[INFO] sample={sample_id} 已在 em_results 中，跳过 EM")
            return

        print(f"[INFO] 运行 EM for sample={sample_id}")
        
        # 构造 MLID 命令
        # ./mlid -i taxident_out -M BEfull -o out_prefix
        
        # 临时输出文件前缀 (mlid 会自动添加 _out.txt)
        mlid_out_prefix = self.out_root / f"{sample_id}_mlid"
        mlid_final_out = self.out_root / f"{sample_id}_mlid_out.txt"
        
        cmd = [
            str(self.mlid_path),
            "-i", str(taxident_out),
            "-M", "B",        # Model B with fixed error rate
            "-d",             # Force dense processing
            "-o", str(mlid_out_prefix),
            "-v"
        ]
        
        print(f"[INFO] Running MLID: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] MLID failed for {sample_id}: {e}")
            return

        # 解析结果
        loglik, p_pop = self._parse_mlid_output(mlid_final_out)

        row = {"sample": sample_id, "loglik": loglik}
        row.update({f"Q_{k}": v for k, v in p_pop.items()})

        if self.result_csv.exists():
            df_old = pd.read_csv(self.result_csv)
            # 确保新列也能被包含 (如果新样本有以前没见过的 hap)
            df_new = pd.concat([df_old, pd.DataFrame([row])], ignore_index=True)
        else:
            df_new = pd.DataFrame([row])

        df_new.to_csv(self.result_csv, index=False)
        print(f"[OK] EM 结果写入 {self.result_csv} 中的一行")
        
        # 清理临时文件
        try:
            if mlid_final_out.exists():
                # mlid_final_out.unlink()
                print(f"[INFO] (Debug: 不删除) MLID 输出保留: {mlid_final_out}")
        except OSError:
            pass

        # ✓ EM 成功后删除 TaxIdent 大文件
        if delete_taxident_after:
            try:
                taxident_out.unlink()
                print(f"[INFO] 删除 TaxIdent 输出: {taxident_out}")
            except OSError:
                pass

    def aggregate_to_population(self, ref_samples_csv: str, output_csv: str):
        """
        读取 hap-level 的 em_results.csv (self.result_csv)，
        结合 ref_samples.csv (mapping: sample -> pop)
        将 proportion 汇总到 Population 级别，并写入 output_csv。
        
        ref_samples.csv 结构期望:
          sample, population (or pop)
          
        em_results.csv 结构:
          sample, loglik, Q_hap1, Q_hap2, ...
        """
        if not self.result_csv.exists():
            print(f"[WARNING] No EM results found at {self.result_csv}")
            return

        # 1. 读取 EM 结果
        df_em = pd.read_csv(self.result_csv)
        
        # 2. 读取 reference mapping
        if not Path(ref_samples_csv).exists():
             print(f"[WARNING] Reference samples CSV not found: {ref_samples_csv}")
             return
        df_ref = pd.read_csv(ref_samples_csv)
        
        # 处理列名可能为 'population' 或 'pop'
        pop_col = "pop"
        if "population" in df_ref.columns:
            pop_col = "population"
        elif "pop" in df_ref.columns:
            pop_col = "pop"
        else:
            raise ValueError(f"{ref_samples_csv} must contain 'sample' and 'population' (or 'pop') columns")
            
        if "sample" not in df_ref.columns:
            raise ValueError(f"{ref_samples_csv} must contain 'sample' column")

        # 建立 sample -> pop 的映射字典
        # ref_samples.csv 中的 sample 是 "tsk_0", "tsk_1" 等
        sample_to_pop = dict(zip(df_ref["sample"].astype(str), df_ref[pop_col].astype(str)))
        
        print(f"[DEBUG] sample_to_pop size: {len(sample_to_pop)}")
        if len(sample_to_pop) > 0:
            print(f"[DEBUG] First 5 samples in ref_samples: {list(sample_to_pop.keys())[:5]}")
        
        # 3. 准备汇总结果
        # 我们需要保留 sample 和 loglik 列，然后把剩下的 Q_ 列聚合
        meta_cols = ["sample", "loglik"]
        q_cols = [c for c in df_em.columns if c.startswith("Q_")]
        
        print(f"[DEBUG] EM results columns (first 5 Q_ cols): {q_cols[:5]}")
        
        # 初始化结果列表
        aggregated_rows = []
        
        # 确定所有可能的 pop 列表
        all_pops = sorted(list(set(sample_to_pop.values())))
        print(f"[DEBUG] All populations found: {all_pops}")
        
        for _, row in df_em.iterrows():
            new_row = {c: row[c] for c in meta_cols if c in row}
            
            # 初始化当前样本的 pop 计数为 0
            pop_counts = {pop: 0.0 for pop in all_pops}
            
            for col in q_cols:
                val = row[col]
                if pd.isna(val):
                    val = 0.0
                
                # col 是 "Q_hapID", 例如 "Q_tsk_0_hap1"
                hap_id = col[2:] 
                
                # 尝试从 hap_id 解析出 sample_id
                if "_hap" in hap_id:
                     sample_id_from_hap = hap_id.rsplit("_hap", 1)[0]
                else:
                     sample_id_from_hap = hap_id # Fallback
                
                # 查找这个 sample 属于哪个 pop
                if sample_id_from_hap in sample_to_pop:
                    pop = sample_to_pop[sample_id_from_hap]
                    pop_counts[pop] += val
                else:
                    # 如果找不到映射 (可能是不在 ref_samples 中的 test sample，或者命名不匹配)
                    # print(f"[DEBUG] Sample {sample_id_from_hap} not in ref samples")
                    pass
            
            # 将聚合后的 pop 比例加入 new_row
            for pop in all_pops:
                new_row[f"Q_{pop}"] = pop_counts[pop]
                
            aggregated_rows.append(new_row)
            
        # 4. 转为 DataFrame 并输出
        df_agg = pd.DataFrame(aggregated_rows)
        
        # 填充可能的 NaN 为 0
        df_agg = df_agg.fillna(0.0)
        
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_agg.to_csv(output_path, index=False)
        print(f"[OK] Aggregated population results written to {output_path}")