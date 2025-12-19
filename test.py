#!/usr/bin/env python

import msprime


def admix_demography_via_migration():
    """
    三源 (AFR, EU, ASN) + 一个 admixed 群体 ADMIX 的模型，
    参数全部来自你截图里的那张表，只是不用 add_admixture，
    而是用 0~12 代之间的一段 migration pulse 近似一次 admixture。

    Pop sizes:
      ANC   =  7310
      AFR   = 14474
      OOA   =  1861
      EU    =  1032   (growth 0.0038 / gen)
      ASN   =   554   (growth 0.0048 / gen)
      ADMIX = 30000   (这里不再给 growth_rate，避免古代 size→0 的问题）

    Times (generations, backwards):
      0–12  : ADMIX <-> (AFR, EU, ASN) 高迁移，近似最近一次 admixture
      12    : 停止 ADMIX <-> 源群体的迁移
      920   : EU–ASN split from OOA
      2040  : AFR–OOA split from ANC

    Base migration (per gen):
      AFR–OOA : 15   × 1e-5
      AFR–EU  : 2.5  × 1e-5
      AFR–ASN : 0.78 × 1e-5
      EU–ASN  : 3.11 × 1e-5
    """

    dem = msprime.Demography()

    # ---------- 现存种群 ----------
    dem.add_population(name="AFR",   initial_size=14474)
    dem.add_population(name="EU",    initial_size=1032, growth_rate=0.0038)
    dem.add_population(name="ASN",   initial_size=554,  growth_rate=0.0048)
    # ADMIX 这次不要 growth_rate，保持大小恒定，避免远古 size 数值趋近 0
    dem.add_population(name="ADMIX", initial_size=30000)

    # ---------- 祖先种群 ----------
    dem.add_population(name="OOA", initial_size=1861)
    dem.add_population(name="ANC", initial_size=7310)

    # ---------- 深层结构：split ----------
    # 920 代：OOA → (EU, ASN)
    dem.add_population_split(
        time=920,
        derived=["EU", "ASN"],
        ancestral="OOA",
    )

    # 2040 代：ANC → (AFR, OOA)
    dem.add_population_split(
        time=2040,
        derived=["AFR", "OOA"],
        ancestral="ANC",
    )

    # ---------- 基础对称迁移（AFR, OOA, EU, ASN 之间） ----------
    dem.set_symmetric_migration_rate(["AFR", "OOA"], 15e-5)
    dem.set_symmetric_migration_rate(["AFR", "EU"],  2.5e-5)
    dem.set_symmetric_migration_rate(["AFR", "ASN"], 0.78e-5)
    dem.set_symmetric_migration_rate(["EU",  "ASN"], 3.11e-5)

    # ---------- 用 migration pulse 近似 admixture ----------
    # 这里我们设定：在 0~12 代之间，ADMIX <-> 源群体有很大的迁移率，
    # 这样往回追溯时，ADMIX 的谱系会很快“跳回” AFR/EU/ASN 中，
    # 起到类似一次 12 代前 admixture 的效果。
    #
    # 目标比例 AFR:EU:ASN = 1/6 : 1/3 : 1/2
    w_AFR, w_EU, w_ASN = 1/6, 1/3, 1/2
    m_total = 0.05  # 总“混合强度”，可以以后慢慢调，这里先给个中等偏大的值

    dem.set_migration_rate("ADMIX", "AFR", m_total * w_AFR)
    dem.set_migration_rate("AFR",   "ADMIX", m_total * w_AFR)

    dem.set_migration_rate("ADMIX", "EU", m_total * w_EU)
    dem.set_migration_rate("EU",    "ADMIX", m_total * w_EU)

    dem.set_migration_rate("ADMIX", "ASN", m_total * w_ASN)
    dem.set_migration_rate("ASN",   "ADMIX", m_total * w_ASN)

    # 在 time=12 把这些 ADMIX <-> 源群体的迁移全部关掉
    for src, dst in [
        ("ADMIX", "AFR"), ("AFR", "ADMIX"),
        ("ADMIX", "EU"),  ("EU",  "ADMIX"),
        ("ADMIX", "ASN"), ("ASN", "ADMIX"),
    ]:
        dem.add_migration_rate_change(time=12, source=src, dest=dst, rate=0.0)

    dem.sort_events()
    dem.validate()
    return dem


if __name__ == "__main__":
    dem = admix_demography_via_migration()
    print(dem.debug())

    # 现代样本：全部在 time=0 采样
    samples = [
        msprime.SampleSet(20, population="AFR",   time=0),
        msprime.SampleSet(20, population="EU",    time=0),
        msprime.SampleSet(20, population="ASN",   time=0),
        msprime.SampleSet(20, population="ADMIX", time=0),
    ]

    ts = msprime.sim_ancestry(
        samples=samples,
        demography=dem,
        sequence_length=1e6,
        recombination_rate=1e-8,
        model="dtwf",        # dtwf 对这种带 admixture 的近代模型更稳
        random_seed=42,
    )

    print("Simulated tree sequence:")
    print(ts)
