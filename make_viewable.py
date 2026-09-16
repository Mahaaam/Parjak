"""
Convert all DICOM studies under ./DICOM into viewable JPEG + catalog + HTML viewer.
Output: ./VIEW/
"""
from __future__ import annotations

import json
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut

ROOT = Path(__file__).resolve().parent
DICOM_ROOT = ROOT / "DICOM"
OUT_ROOT = ROOT / "VIEW"
IMG_ROOT = OUT_ROOT / "img"


def _as_float(value) -> float | None:
    if value is None:
        return None
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def render_frame(ds: pydicom.Dataset, arr: np.ndarray) -> Image.Image:
    """Render one 2D frame to 8-bit grayscale using DICOM VOI when possible."""
    if arr.ndim > 2:
        raise ValueError(f"Expected 2D frame, got {arr.shape}")

    try:
        scaled = apply_modality_lut(arr, ds)
    except Exception:
        slope = _as_float(getattr(ds, "RescaleSlope", 1)) or 1.0
        intercept = _as_float(getattr(ds, "RescaleIntercept", 0)) or 0.0
        scaled = arr.astype(np.float32) * slope + intercept

    try:
        windowed = apply_voi_lut(scaled, ds)
    except Exception:
        wc = _as_float(getattr(ds, "WindowCenter", None))
        ww = _as_float(getattr(ds, "WindowWidth", None))
        if wc is not None and ww is not None and ww > 0:
            lo = wc - ww / 2.0
            hi = wc + ww / 2.0
            windowed = np.clip(scaled.astype(np.float32), lo, hi)
        else:
            windowed = scaled.astype(np.float32)

    windowed = np.asarray(windowed, dtype=np.float32)
    lo = float(np.min(windowed))
    hi = float(np.max(windowed))
    if hi <= lo:
        out = np.zeros_like(windowed, dtype=np.uint8)
    else:
        out = ((windowed - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)

    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
    if photometric == "MONOCHROME1":
        out = 255 - out
    return Image.fromarray(out)


def safe_name(text: str, fallback: str = "unknown") -> str:
    text = (text or fallback).strip()
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        elif ch in (" ", "/", "\\", ":", "^"):
            keep.append("_")
    name = "".join(keep).strip("_")
    return (name[:80] or fallback)


def read_meta(path: Path) -> dict | None:
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception:
        return None

    modality = str(getattr(ds, "Modality", "OT") or "OT")
    # Skip non-image docs when no Rows
    if modality in {"SR", "PR", "KO"} and not hasattr(ds, "Rows"):
        return {
            "path": str(path),
            "modality": modality,
            "skip_pixels": True,
            "study_date": str(getattr(ds, "StudyDate", "") or ""),
            "study_time": str(getattr(ds, "StudyTime", "") or "")[:6],
            "study_uid": str(getattr(ds, "StudyInstanceUID", "") or ""),
            "series_uid": str(getattr(ds, "SeriesInstanceUID", "") or ""),
            "series_desc": str(getattr(ds, "SeriesDescription", "") or modality),
            "study_desc": str(getattr(ds, "StudyDescription", "") or ""),
            "series_number": str(getattr(ds, "SeriesNumber", "") or ""),
            "instance_number": str(getattr(ds, "InstanceNumber", "") or ""),
            "sop_uid": str(getattr(ds, "SOPInstanceUID", "") or ""),
            "frames": 0,
        }

    frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    return {
        "path": str(path),
        "modality": modality,
        "skip_pixels": False,
        "study_date": str(getattr(ds, "StudyDate", "") or ""),
        "study_time": str(getattr(ds, "StudyTime", "") or "")[:6],
        "study_uid": str(getattr(ds, "StudyInstanceUID", "") or ""),
        "series_uid": str(getattr(ds, "SeriesInstanceUID", "") or ""),
        "series_desc": str(getattr(ds, "SeriesDescription", "") or modality),
        "study_desc": str(getattr(ds, "StudyDescription", "") or ""),
        "series_number": str(getattr(ds, "SeriesNumber", "") or ""),
        "instance_number": str(getattr(ds, "InstanceNumber", "") or "0"),
        "sop_uid": str(getattr(ds, "SOPInstanceUID", "") or ""),
        "rows": int(getattr(ds, "Rows", 0) or 0),
        "cols": int(getattr(ds, "Columns", 0) or 0),
        "frames": frames,
        "patient_name": str(getattr(ds, "PatientName", "") or ""),
        "patient_id": str(getattr(ds, "PatientID", "") or ""),
        "patient_sex": str(getattr(ds, "PatientSex", "") or ""),
        "patient_age": str(getattr(ds, "PatientAge", "") or ""),
        "body_part": str(getattr(ds, "BodyPartExamined", "") or ""),
    }


def convert_file(meta: dict, series_dir: Path, counter: list[int]) -> list[dict]:
    if meta.get("skip_pixels"):
        return []

    path = Path(meta["path"])
    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array
    outputs: list[dict] = []
    samples = int(getattr(ds, "SamplesPerPixel", 1) or 1)
    frames_meta = int(getattr(ds, "NumberOfFrames", 1) or 1)

    # RGB/RGBA still image: (H, W, 3|4) — NOT multi-frame
    if arr.ndim == 3 and arr.shape[-1] in (3, 4) and (samples in (3, 4) or frames_meta == 1):
        frames = [arr]
    elif arr.ndim == 2:
        frames = [arr]
    elif arr.ndim == 3:
        frames = [arr[i] for i in range(arr.shape[0])]
    elif arr.ndim == 4:
        # (frames, H, W, channels)
        frames = [arr[i] for i in range(arr.shape[0])]
    else:
        raise ValueError(f"Unsupported shape {arr.shape} for {path}")

    inst = meta.get("instance_number") or "0"
    for fi, frame in enumerate(frames):
        counter[0] += 1
        fname = f"{counter[0]:07d}_i{inst}_f{fi + 1}.jpg"
        out_path = series_dir / fname
        if frame.ndim == 3 and frame.shape[-1] in (3, 4):
            # Color DICOM (e.g. radial MPR capture)
            img = Image.fromarray(frame[:, :, :3].astype(np.uint8))
        else:
            img = render_frame(ds, frame)
        img.save(out_path, format="JPEG", quality=90, optimize=True)
        rel = out_path.relative_to(OUT_ROOT).as_posix()
        outputs.append(
            {
                "file": rel,
                "instance_number": inst,
                "frame": fi + 1,
                "source": str(path.relative_to(ROOT)).replace("\\", "/"),
                "rows": int(frame.shape[0]),
                "cols": int(frame.shape[1]),
            }
        )
    return outputs


def build_html(catalog: dict) -> str:
    return """<!doctype html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Parjak DICOM Viewer</title>
<style>
:root { --bg:#0f1419; --panel:#1a222c; --text:#e8eef5; --muted:#9aa7b5; --accent:#3d9cf0; }
* { box-sizing:border-box; }
body { margin:0; font-family:Tahoma,Segoe UI,sans-serif; background:var(--bg); color:var(--text); height:100vh; display:flex; }
#sidebar { width:360px; max-width:40vw; background:var(--panel); overflow:auto; border-left:1px solid #2a3542; padding:12px; }
#main { flex:1; display:flex; flex-direction:column; min-width:0; }
#toolbar { padding:10px 14px; background:#141b23; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
#stage { flex:1; display:flex; align-items:center; justify-content:center; overflow:auto; background:#000; }
#stage img { max-width:100%; max-height:100%; object-fit:contain; }
h1 { font-size:16px; margin:0 0 10px; }
.study { margin-bottom:14px; }
.study h2 { font-size:13px; margin:0 0 6px; color:var(--accent); }
.series { display:block; width:100%; text-align:right; background:transparent; border:1px solid #2e3b4a; color:var(--text); padding:8px; margin:0 0 6px; border-radius:6px; cursor:pointer; }
.series:hover, .series.active { border-color:var(--accent); background:#223041; }
.meta { color:var(--muted); font-size:12px; }
button.nav { background:#243140; color:var(--text); border:1px solid #3a4b5e; border-radius:6px; padding:6px 12px; cursor:pointer; }
#slider { width:min(420px,60%); }
</style>
</head>
<body>
<aside id="sidebar"><h1>همه تصاویر DICOM</h1><div id="list"></div></aside>
<section id="main">
  <div id="toolbar">
    <button class="nav" id="prev">قبلی</button>
    <button class="nav" id="next">بعدی</button>
    <input type="range" id="slider" min="0" max="0" value="0"/>
    <span class="meta" id="info">سری را انتخاب کنید</span>
  </div>
  <div id="stage"><img id="view" alt="dicom"/></div>
</section>
<script src="catalog.js"></script>
<script>
const listEl = document.getElementById('list');
const view = document.getElementById('view');
const info = document.getElementById('info');
const slider = document.getElementById('slider');
let images = [];
let idx = 0;
let activeBtn = null;

function show(i) {
  if (!images.length) return;
  idx = Math.max(0, Math.min(images.length - 1, i));
  view.src = images[idx].file;
  slider.value = idx;
  info.textContent = (idx + 1) + ' / ' + images.length + ' | frame ' + images[idx].frame + ' | ' + (images[idx].source || '');
}

function loadSeries(series, btn) {
  images = series.images || [];
  if (activeBtn) activeBtn.classList.remove('active');
  activeBtn = btn;
  btn.classList.add('active');
  slider.max = Math.max(0, images.length - 1);
  show(0);
}

(catalog.studies || []).forEach(study => {
  const box = document.createElement('div');
  box.className = 'study';
  const h = document.createElement('h2');
  h.textContent = (study.study_date || '') + ' | ' + (study.modality || '') + ' | ' + (study.study_desc || '');
  box.appendChild(h);
  const m = document.createElement('div');
  m.className = 'meta';
  m.textContent = (study.patient_name || '') + ' | ID ' + (study.patient_id || '');
  box.appendChild(m);
  (study.series || []).forEach(series => {
    const b = document.createElement('button');
    b.className = 'series';
    b.textContent = '#' + (series.series_number || '?') + ' ' + (series.series_desc || '') + ' (' + (series.images || []).length + ')';
    b.onclick = () => loadSeries(series, b);
    box.appendChild(b);
  });
  listEl.appendChild(box);
});

document.getElementById('prev').onclick = () => show(idx - 1);
document.getElementById('next').onclick = () => show(idx + 1);
slider.oninput = () => show(Number(slider.value));
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown') show(idx + 1);
  if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') show(idx - 1);
});
</script>
</body>
</html>
"""


def main() -> int:
    if not DICOM_ROOT.exists():
        print("DICOM folder not found:", DICOM_ROOT)
        return 1

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    IMG_ROOT.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in DICOM_ROOT.rglob("*") if p.is_file())
    print(f"Found {len(files)} files under DICOM")

    metas: list[dict] = []
    for i, path in enumerate(files, 1):
        meta = read_meta(path)
        if meta is None:
            print("SKIP unreadable:", path)
            continue
        metas.append(meta)
        if i % 200 == 0:
            print(f"  indexed {i}/{len(files)}")

    # Group study -> series -> instances
    studies: dict[str, dict] = {}
    for meta in metas:
        study_key = meta["study_uid"] or f"{meta['study_date']}_{meta['study_time']}"
        series_key = meta["series_uid"] or f"{study_key}_{meta['series_number']}"
        study = studies.setdefault(
            study_key,
            {
                "study_uid": meta["study_uid"],
                "study_date": meta["study_date"],
                "study_time": meta["study_time"],
                "study_desc": meta["study_desc"],
                "patient_name": meta.get("patient_name", ""),
                "patient_id": meta.get("patient_id", ""),
                "patient_sex": meta.get("patient_sex", ""),
                "patient_age": meta.get("patient_age", ""),
                "modality": meta["modality"],
                "series_map": {},
            },
        )
        if meta["modality"] not in (study["modality"] or ""):
            if study["modality"] and meta["modality"] not in study["modality"]:
                study["modality"] = f"{study['modality']}/{meta['modality']}"
            elif not study["modality"]:
                study["modality"] = meta["modality"]

        series = study["series_map"].setdefault(
            series_key,
            {
                "series_uid": meta["series_uid"],
                "series_number": meta["series_number"],
                "series_desc": meta["series_desc"],
                "modality": meta["modality"],
                "body_part": meta.get("body_part", ""),
                "instances": [],
            },
        )
        series["instances"].append(meta)

    catalog = {
        "patient_name": "",
        "patient_id": "",
        "total_source_files": len(metas),
        "studies": [],
    }
    counter = [0]
    errors: list[str] = []

    for study_key, study in sorted(
        studies.items(), key=lambda kv: (kv[1].get("study_date") or "", kv[1].get("study_time") or "")
    ):
        if not catalog["patient_name"]:
            catalog["patient_name"] = study.get("patient_name", "")
            catalog["patient_id"] = study.get("patient_id", "")

        study_out = {
            "study_uid": study["study_uid"],
            "study_date": study["study_date"],
            "study_time": study["study_time"],
            "study_desc": study["study_desc"],
            "patient_name": study.get("patient_name", ""),
            "patient_id": study.get("patient_id", ""),
            "modality": study.get("modality", ""),
            "series": [],
        }

        date_part = study["study_date"] or "unknown_date"
        for series_key, series in sorted(
            study["series_map"].items(),
            key=lambda kv: (int(kv[1]["series_number"]) if str(kv[1]["series_number"]).isdigit() else 9999, kv[1]["series_desc"]),
        ):
            series_name = f"{date_part}_{safe_name(series['modality'])}_s{safe_name(series['series_number'], 'x')}_{safe_name(series['series_desc'])}"
            series_dir = IMG_ROOT / series_name
            series_dir.mkdir(parents=True, exist_ok=True)

            # Sort instances
            insts = sorted(
                series["instances"],
                key=lambda m: (
                    int(m["instance_number"]) if str(m["instance_number"]).isdigit() else 0,
                    m["path"],
                ),
            )
            images: list[dict] = []
            for meta in insts:
                try:
                    images.extend(convert_file(meta, series_dir, counter))
                except Exception as exc:
                    msg = f"{meta['path']}: {exc}"
                    errors.append(msg)
                    print("ERR", msg)

            study_out["series"].append(
                {
                    "series_uid": series["series_uid"],
                    "series_number": series["series_number"],
                    "series_desc": series["series_desc"],
                    "modality": series["modality"],
                    "body_part": series.get("body_part", ""),
                    "images": images,
                }
            )
            print(
                f"OK {study['study_date']} | {series['modality']} | "
                f"#{series['series_number']} {series['series_desc']}: {len(images)} images"
            )

        catalog["studies"].append(study_out)

    catalog["total_images"] = counter[0]
    catalog["errors"] = errors

    (OUT_ROOT / "catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_ROOT / "catalog.js").write_text(
        "var catalog = " + json.dumps(catalog, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    (OUT_ROOT / "index.html").write_text(build_html(catalog), encoding="utf-8")

    print("\nDONE")
    print("images:", counter[0])
    print("errors:", len(errors))
    print("open:", OUT_ROOT / "index.html")
    return 0 if not errors else 0  # still usable with partial errors


if __name__ == "__main__":
    raise SystemExit(main())
