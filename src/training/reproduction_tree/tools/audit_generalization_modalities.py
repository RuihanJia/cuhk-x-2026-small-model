"""Descriptive audit of existing F0/F3 outputs and a fixed Train-only schema sample.

No model fitting, new model inference, Test reads, or held-out-fold selection.
Oracle counts use known development labels and are diagnostic ceilings only.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/generalization_review_20260910_v1"
SOURCES = {}
FOLD_USERS = {0: {2, 16, 20}, 3: {5, 7, 8, 21}}


def identity(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    key = p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)
    if key not in SOURCES:
        SOURCES[key] = {
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest().upper(),
        }
    return p


def read_json(path):
    return json.loads(identity(path).read_text(encoding="utf-8-sig"))


def load_npz(path):
    with np.load(identity(path), allow_pickle=False) as d:
        out = {k: d[k].copy() for k in d.files}
    if "labels" in out:
        out["targets"] = out["labels"]
    assert len(set(out["clip_ids"].tolist())) == len(out["clip_ids"])
    return out


def metrics(y, logits, users):
    assert logits.shape == (len(y), 40) and np.isfinite(logits).all()
    pred = logits.argmax(1)
    correct = pred == y
    cm = np.bincount(y * 40 + pred, minlength=1600).reshape(40, 40)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * cm.diagonal(), denom, out=np.zeros(40), where=denom > 0)
    per_user = {
        str(int(u)): {"n": int((users == u).sum()), "correct": int(correct[users == u].sum()),
                      "accuracy": float(correct[users == u].mean())}
        for u in np.unique(users)
    }
    return {"n": len(y), "correct": int(correct.sum()), "accuracy": float(correct.mean()),
            "macro_f1_fixed_40": float(f1.mean()), "per_user": per_user,
            "user_equal_accuracy": float(np.mean([v["accuracy"] for v in per_user.values()]))}


def pair(y, a, b, users):
    pa, pb = a.argmax(1), b.argmax(1)
    ca, cb = pa == y, pb == y
    return {
        "n": len(y), "a_correct": int(ca.sum()), "b_correct": int(cb.sum()),
        "b_correct_a_wrong": int((~ca & cb).sum()),
        "a_correct_b_wrong": int((ca & ~cb).sum()),
        "both_correct": int((ca & cb).sum()), "both_wrong": int((~ca & ~cb).sum()),
        "prediction_disagreement": int((pa != pb).sum()),
        "oracle_union_correct": int((ca | cb).sum()),
        "oracle_union_accuracy": float((ca | cb).mean()),
        "per_user": {str(int(u)): {
            "n": int((users == u).sum()),
            "a_correct": int(ca[users == u].sum()), "b_correct": int(cb[users == u].sum()),
            "b_correct_a_wrong": int((~ca[users == u] & cb[users == u]).sum()),
            "a_correct_b_wrong": int((ca[users == u] & ~cb[users == u]).sum()),
        } for u in np.unique(users)},
    }


def aligned(ref, other):
    lookup = {s: i for i, s in enumerate(other["clip_ids"].tolist())}
    ia = np.array([i for i, s in enumerate(ref["clip_ids"].tolist()) if s in lookup], dtype=int)
    ib = np.array([lookup[s] for s in ref["clip_ids"][ia].tolist()], dtype=int)
    assert np.array_equal(ref["targets"][ia], other["targets"][ib])
    assert np.array_equal(ref["users"][ia], other["users"][ib])
    return ia, ib


def validate_membership(d, fold, manifest):
    assert set(d["users"].tolist()) <= FOLD_USERS[fold]
    for s, y, u in zip(d["clip_ids"].tolist(), d["targets"].tolist(), d["users"].tolist()):
        assert s in manifest and manifest[s]["label"] == y and manifest[s]["user"] == u
    if "valid" in d:
        assert d["valid"].all(), "Missing branch inputs must not be counted as valid predictions"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in identity("data/manifests/train_manifest.jsonl").read_text(
        encoding="utf-8-sig").splitlines() if x.strip()]
    manifest = {r["clip_id"]: r for r in rows}
    assert len(manifest) == len(rows) == 3036
    modalities = ["Depth_Color", "IR", "Thermal", "Skeleton", "IMU", "Radar"]
    result = {
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Descriptive recomputation of existing development F0/F3 outputs; no selection or new fits",
        "oracle_warning": "Uses held-out development labels; unattainable oracle routing, not a realizable score or Private forecast",
        "training_manifest": {
            "n": len(rows), "users": sorted({r["user"] for r in rows}),
            "path_present_counts_not_signal_validity": {
                m: sum(bool(r["modalities"].get(m)) for r in rows) for m in modalities},
            "all_six_paths_present_not_signal_validity": sum(
                all(r["modalities"].get(m) for m in modalities) for r in rows),
        }, "folds": {}, "pooled": {},
    }
    sensor_cache = load_npz("archive/20260910_handoff/project/runs/research_series_20260909_v2/bundle/data/measured_sensors.npz")
    result["current_paired_sensor_cache"] = {
        "n": len(sensor_cache["clip_ids"]), "imu_valid": int(sensor_cache["imu_valid"].sum()),
        "imu_valid_per_user": {str(int(u)): {
            "n": int((sensor_cache["users"] == u).sum()),
            "valid": int(sensor_cache["imu_valid"][sensor_cache["users"] == u].sum())
        } for u in np.unique(sensor_cache["users"])},
    }
    protocol = read_json("runs/s4_tail_average_20260910_v1/bundle/protocol.json")
    thermal_audit = read_json("archive/20260907_handoff/project/runs/_cloud/20260907-multimodal-thermal/independent_acceptance/THERMAL_INDEPENDENT_ACCEPTANCE.json")
    radar_audit = read_json("archive/20260907_handoff/project/runs/20260907-local-radar-mhomeges-inner-formal-v1/RADAR_INDEPENDENT_ACCEPTANCE.json")
    result["radar_prior_verified_coverage"] = radar_audit["coverage"]
    pooled = collections.defaultdict(list)
    aux_native = collections.defaultdict(list)
    aux_paired = collections.defaultdict(list)
    all_oracle_rows = []
    for fold in [0, 3]:
        ref = load_npz(f"runs/s4_tail_average_20260910_v1/bundle/data/reference_f{fold}.npz")
        video = load_npz(f"runs/d5_spatiotemporal_grid_20260910_v1/output/D5_F{fold}_original/predictions.npz")
        validate_membership(ref, fold, manifest)
        ia, ib = aligned(ref, video)
        assert len(ia) == len(ref["clip_ids"]) == len(video["clip_ids"])
        assert not set(ref["users"].tolist()) & set(protocol["assets"][f"mvit_f{fold}"]["train_users"])
        z = video["cross_five_logits"][ib]
        sensor = ref["sensor"]
        fused = .8 * z + .2 * sensor
        assert np.array_equal(.8 * ref["original_video"] + .2 * sensor, ref["original_fused"])
        y, users = ref["targets"], ref["users"]
        poolrow = {"targets": y, "users": users, "video": z, "sensor": sensor, "fused": fused}
        for k, v in poolrow.items(): pooled[k].append(v)
        f = {"video": metrics(y, z, users), "sensor_joint": metrics(y, sensor, users),
             "fused": metrics(y, fused, users),
             "video_vs_sensor": pair(y, z, sensor, users),
             "video_vs_fixed_fused": pair(y, z, fused, users), "auxiliary": {}}
        r1_correct = fused.argmax(1) == y
        branch_correct = {"sensor_joint": sensor.argmax(1) == y}
        branch_union = r1_correct | branch_correct["sensor_joint"]
        for name, relative in {
            "thermal_x3d": f"archive/20260907_handoff/project/runs/_cloud/20260907-multimodal-thermal/independent_acceptance/fold{fold}_independent_heldout_fp32.npz",
            "imu_control": f"archive/20260907_handoff/project/runs/20260907-local-imu-orientation-formal-v1/science/F{fold}/control/HELDOUT_LOGITS.npz",
            "imu_orientation": f"archive/20260907_handoff/project/runs/20260907-local-imu-orientation-formal-v1/science/F{fold}/orientation/HELDOUT_LOGITS.npz",
            "skeleton_top1": f"archive/20260907_handoff/project/runs/20260907-local-skeleton-person-retention-formal-v1/science/F{fold}/top1/HELDOUT_LOGITS.npz",
            "skeleton_top4": f"archive/20260907_handoff/project/runs/20260907-local-skeleton-person-retention-formal-v1/science/F{fold}/top4/HELDOUT_LOGITS.npz",
            "radar_native": f"archive/20260907_handoff/project/runs/20260907-local-radar-mhomeges-inner-formal-v1/science/F{fold}/native_only/heldout_features_logits_fp32.npz",
        }.items():
            d = load_npz(relative)
            validate_membership(d, fold, manifest)
            if name.startswith(("imu_", "skeleton_")):
                fit = read_json(Path(relative).with_name("FIT_RESULT.json"))
                assert not set(fit["train_users"]) & set(fit["check_users"])
                assert set(fit["check_users"]) == FOLD_USERS[fold]
                assert fit["heldout_labels_or_metrics_seen_during_fit"] is False
            elif name == "thermal_x3d":
                audit = thermal_audit["folds"][str(fold)]
                assert audit["heldout_features_logits_fp32_sha256"] == SOURCES[relative]["sha256"]
                assert set(audit["heldout_users"]) == FOLD_USERS[fold]
            i, j = aligned(ref, d)
            f["auxiliary"][name] = {
                "source": relative, "native_eligible": metrics(d["targets"], d["logits"], d["users"]),
                "paired_with_current_video": pair(y[i], z[i], d["logits"][j], users[i]),
                "paired_with_current_R1_fused": pair(y[i], fused[i], d["logits"][j], users[i]),
                "outside_current_video_coverage": len(d["clip_ids"]) - len(j),
            }
            aux_native[name].append((d["targets"], d["logits"], d["users"]))
            aux_paired[name].append((y[i], z[i], fused[i], d["logits"][j], users[i]))
            if name in ["thermal_x3d", "imu_orientation", "skeleton_top4", "radar_native"]:
                bc = np.zeros(len(y), dtype=bool)
                bc[i] = d["logits"][j].argmax(1) == y[i]
                branch_correct[name] = bc
                branch_union |= bc
        all_oracle_rows.append((r1_correct, branch_union, branch_correct, users))
        result["folds"][str(fold)] = f
    p = {k: np.concatenate(v) for k, v in pooled.items()}
    result["pooled"].update({k: metrics(p["targets"], p[k], p["users"]) for k in ["video", "sensor", "fused"]})
    result["pooled"]["video_vs_sensor"] = pair(p["targets"], p["video"], p["sensor"], p["users"])
    result["pooled"]["video_vs_fixed_fused"] = pair(p["targets"], p["video"], p["fused"], p["users"])
    result["pooled"]["auxiliary"] = {}
    for name, parts in aux_native.items():
        y, z, u = (np.concatenate(v) for v in zip(*parts))
        py, pv, pf, pz, pu = (np.concatenate(v) for v in zip(*aux_paired[name]))
        result["pooled"]["auxiliary"][name] = {
            "native_eligible": metrics(y, z, u),
            "paired_with_current_video": pair(py, pv, pz, pu),
            "paired_with_current_R1_fused": pair(py, pf, pz, pu),
        }
    r1 = np.concatenate([r[0] for r in all_oracle_rows])
    union = np.concatenate([r[1] for r in all_oracle_rows])
    bcs = {name: np.concatenate([r[2][name] for r in all_oracle_rows]) for name in all_oracle_rows[0][2]}
    result["pooled"]["auxiliary_oracle"] = {
        "base_R1_correct": int(r1.sum()), "n": len(r1),
        "any_selected_auxiliary_correct_on_R1_error": int((~r1 & union).sum()),
        "union_correct": int(union.sum()),
        "exclusive_recovery_among_selected_auxiliaries": {
            name: int((~r1 & bc & ~np.logical_or.reduce([x for k, x in bcs.items() if k != name])).sum())
            for name, bc in bcs.items()},
        "caution": "Historical branch comparison; zero for unavailable rows only in correctness masks. No missing logits treated as predictions. No trained fusion or performance claim.",
    }
    # Fixed source/schema sample: first existing clip per class and each of four fitting users.
    selected = []
    for u in [4, 6, 17, 23]:
        for label in range(40):
            candidates = [r for r in rows if r["user"] == u and r["label"] == label and r["modalities"].get("IMU")]
            if candidates:
                selected.append(sorted(candidates, key=lambda r: r["clip_id"])[0])
    schema = {"scope": "First Train clip per class for fitting users4,6,17,23; no model evaluation; finite/nonconstant does not verify timing or calibration",
              "selected_clip_ids": [r["clip_id"] for r in selected], "n": len(selected), "by_device": {}}
    device_counts = collections.defaultdict(collections.Counter)
    for r in selected:
        for raw in sorted(Path(r["modalities"]["IMU"]).glob("*.csv")):
            groups = collections.defaultdict(list)
            with identity(raw).open(encoding="utf-8-sig") as handle:
                for sample in csv.DictReader(handle):
                    device = sample.get("设备名称", "").split("(")[0]
                    groups[device].append(sample)
            for device, samples in groups.items():
                dc = device_counts[device]
                dc["clips_present"] += 1
                for channel, prefix in [("acceleration", "加速度"), ("gyro", "角速度")]:
                    cols = [k for k in samples[0] if k.startswith(prefix)]
                    valid = len(cols) == 3
                    for col in cols:
                        vals = []
                        for sample in samples:
                            try:
                                value = float(sample[col])
                                if np.isfinite(value): vals.append(value)
                            except (ValueError, TypeError): pass
                        valid = valid and len(vals) >= 8 and len(set(vals)) >= 2
                    if valid: dc[channel + "_three_axes_finite8_nonconstant"] += 1
    schema["by_device"] = dict(device_counts)
    result["raw_IMU_schema_sample"] = schema
    # Bind prior positive/mixed/negative evidence without rewriting historical decisions.
    result["historical_evidence_paths"] = [
        "archive/20260909_handoff/project/runs/sensor_accgyro_20260908/OBSERVATION.json",
        "archive/20260909_handoff/project/protocols/sensor_accgyro_20260908_v1.json",
        "archive/20260909_handoff/project/runs/cloud_wtc_imu_20260908/VERIFIED.json",
        "archive/20260909_handoff/project/runs/local_dsads_transfer_20260908/VERIFIED.json",
        "archive/20260909_handoff/project/runs/mvit_thermal_20260908/verified_F0/VERIFIED.json",
        "archive/20260910_handoff/project/runs/sensor_radar_20260910_v1/LOCAL_INDEPENDENT_FINAL.json",
        "archive/20260907_handoff/project/runs/_engineering/multimodal_start_20260907/SIX_FUSION_INDEPENDENT_ACCEPTANCE.json",
        "archive/20260907_handoff/project/runs/_engineering/local_imu_integration_20260907/INDEPENDENT_ACCEPTANCE_01.json",
        "evidence/rules/RULES_SNAPSHOT_20260904.md",
        "evidence/rules/source_20260904/external_data_host_clarification.png",
        "deliverables/mvit_fiveview_candidate_20260910/raw_sensor.py",
        "deliverables/mvit_fiveview_candidate_20260910/sensor_model.py",
    ]
    for evidence_path in result["historical_evidence_paths"]:
        identity(evidence_path)
    result["new_training"] = False
    result["new_model_inference"] = False
    result["Test_F2_F4_evaluation"] = False
    result["source_identities"] = SOURCES
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper()
    dest = OUT / "AUDIT.json"
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(dest), "primary": result["pooled"]["video_vs_fixed_fused"],
                      "oracle": result["pooled"]["auxiliary_oracle"], "imu_sample": {k:v for k,v in schema.items() if k != "selected_clip_ids"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
