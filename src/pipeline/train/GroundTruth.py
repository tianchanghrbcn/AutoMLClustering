#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import multiprocessing as mp

from src.pipeline.train.cluster_methods import ClusterMethod, run_clustering


def copy_groundtruth(force: bool) -> None:
    here = Path(__file__).resolve().parent
    comp_path = here / "comparison.json"
    root = comp_path.parent
    copied = skipped = missing = 0
    with comp_path.open("r", encoding="utf-8") as fp:
        cfgs = json.load(fp)
    for cfg in cfgs:
        ds_id = cfg["dataset_id"]
        src_rel = Path(cfg["paths"]["clean_csv"])
        src = (root / src_rel).resolve()
        rep_paths = cfg["paths"].setdefault("repaired_paths", {})
        gt_rel = Path(
            rep_paths.get(
                "GroundTruth",
                Path("../../../results/cleaned_data/GroundTruth") / f"repaired_{ds_id}.csv",
            )
        )
        rep_paths["GroundTruth"] = str(gt_rel)
        dst = (root / gt_rel).resolve()
        if not src.exists():
            missing += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not force:
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1
    print(f"GroundTruth 复制完成: 新复制 {copied} 个, 跳过 {skipped} 个, 源缺失 {missing} 个")


def process_record(idx, rec, work_dir, runtime_lut):
    clustered = []
    dataset_id = idx
    error_rate = rec.get("error_rate", -1)
    if abs(error_rate - 0.01) < 1e-12:
        return clustered
    strategies = {9: "GroundTruth"}
    cleaning_rt = runtime_lut.get(dataset_id, 0.0)
    for algo_name in strategies.values():
        rep_path = os.path.join(
            work_dir, "results", "cleaned_data", algo_name, f"repaired_{dataset_id}.csv"
        )
        if not os.path.exists(rep_path):
            continue
        for cid in range(6):
            out_dir, clu_rt = run_clustering(
                dataset_id=dataset_id,
                algorithm=algo_name,
                cluster_method_id=cid,
                cleaned_file_path=rep_path,
            )
            if out_dir and clu_rt:
                clustered.append(
                    dict(
                        dataset_id=dataset_id,
                        cleaning_algorithm=algo_name,
                        cleaning_runtime=cleaning_rt,
                        clustering_algorithm=cid,
                        clustering_name=ClusterMethod(cid).name,
                        clustering_runtime=clu_rt,
                        clustered_file_path=out_dir,
                    )
                )
    return clustered


def cluster_all() -> None:
    work_dir = Path(__file__).resolve().parents[3]
    ev_path = work_dir / "results/eigenvectors.json"
    clu_path = work_dir / "results/clustered_results_groundtruth.json"
    cln_path = work_dir / "results/cleaned_results.json"
    with ev_path.open("r", encoding="utf-8") as f:
        ev_records = json.load(f)
    runtime_lut = {}
    if cln_path.exists():
        with cln_path.open("r", encoding="utf-8") as f:
            runtime_lut = {i["dataset_id"]: i["runtime"] for i in json.load(f)}
    clustered_total = []
    with ProcessPoolExecutor(max_workers=mp.cpu_count(),
                             mp_context=mp.get_context("spawn")) as ex:
        futs = [
            ex.submit(process_record, i, r, str(work_dir), runtime_lut)
            for i, r in enumerate(ev_records)
        ]
        for fut in futs:
            try:
                clustered_total.extend(fut.result())
            except Exception as e:
                print(f"聚类子任务异常: {e}")
    with clu_path.open("w", encoding="utf-8") as f:
        json.dump(clustered_total, f, ensure_ascii=False, indent=4)
    print(f"聚类结果已写入 {clu_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    copy_groundtruth(args.force)
    cluster_all()


if __name__ == "__main__":
    main()
