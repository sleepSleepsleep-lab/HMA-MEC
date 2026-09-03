# -*- coding: utf-8 -*-
"""Milano trace 预处理: 提取每 10 分钟 × 每网格的 Internet 流量时序.
输入: results/milano_data/sms-call-internet-mi-2013-11-{01..07}.csv
输出: results/milano_data/trace_internet.npy  (shape (7*144, 10000), float32)
      results/milano_data/trace_days.npy      (每行的日期/时刻标签)
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "results", "milano_data")

N_CELLS = 10000
STEPS_PER_DAY = 144     # 24h × 6 (每 10 分钟)
DAYS = ["01", "02", "03", "04", "05", "06", "07"]


def main():
    rows = []
    for day in DAYS:
        p = os.path.join(DATA, f"sms-call-internet-mi-2013-11-{day}.csv")
        print(f"  读取 {p} ...", flush=True)
        df = pd.read_csv(p, usecols=["datetime", "CellID", "internet"])
        df["internet"] = pd.to_numeric(df["internet"], errors="coerce").fillna(0.0)
        df["CellID"] = df["CellID"] - 1          # 1-based → 0-based
        # 10 分钟时间槽: 按 datetime 排序并映射到 0..143
        dt = pd.to_datetime(df["datetime"])
        slot = (dt.dt.hour * 60 + dt.dt.minute) // 10
        df["slot"] = slot
        mat = np.zeros((STEPS_PER_DAY, N_CELLS), dtype=np.float32)
        agg = df.groupby(["slot", "CellID"])["internet"].sum()
        for (sl, cid), v in agg.items():
            if 0 <= sl < STEPS_PER_DAY and 0 <= cid < N_CELLS:
                mat[sl, cid] = float(v)
        rows.append(mat)
        print(f"    day {day}: 非零单元 {np.count_nonzero(mat)}/{mat.size} "
              f"总量 {mat.sum():.1f}", flush=True)
    trace = np.concatenate(rows, axis=0)         # (7*144, 10000)
    np.save(os.path.join(DATA, "trace_internet.npy"), trace)
    print(f"  保存 trace_internet.npy: {trace.shape} "
          f"(非零 {(trace > 0).mean():.1%}, 总流量 {trace.sum():.1f})")


if __name__ == "__main__":
    main()