
import os
import glob
import urllib.request
import numpy as np
import nibabel as nib
from nilearn import datasets as nl_datasets, plotting as nl_plotting
from netneurotools import datasets as nnt_datasets


# ---------------- helpers ----------------

def _decode_fs_names(names):
    """Decode FreeSurfer names to str."""
    out = []
    for n in names:
        out.append(n.decode() if isinstance(n, (bytes, bytearray)) else str(n))
    return out


def _fetch_schaefer_annots(mesh='fsaverage5', n_rois=100, yeo_networks=7):
    """
    Return {'lh': path, 'rh': path} to Schaefer2018 .annot files for the given mesh (e.g., fsaverage5).
    Uses netneurotools 0.2.5 legacy cache; if not found, downloads from CBIG and caches under:
      ~/.cache/schaefer2018/<mesh>/label/
    """
    search_roots = []

    # 1) Try legacy nnt fetcher (returns a Bunch/dict of paths)
    try:
        base = nnt_datasets.fetch_schaefer2018(version=mesh)
    except Exception:
        base = None

    if isinstance(base, str) and os.path.isdir(base):
        search_roots.append(base)
    elif hasattr(base, 'items'):
        for _, v in base.items():
            if isinstance(v, str) and os.path.isdir(v):
                search_roots.append(v)
            elif isinstance(v, (list, tuple)):
                for t in v:
                    if isinstance(t, str) and os.path.isdir(t):
                        search_roots.append(t)

    # 2) Default legacy cache root
    search_roots.append(os.path.join(os.path.expanduser('~'), 'netneurotools_data'))

    # Look for the expected filenames
    pat_lh = f"*{n_rois}Parcels*{yeo_networks}Networks*lh.annot"
    pat_rh = f"*{n_rois}Parcels*{yeo_networks}Networks*rh.annot"
    lh = rh = None
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        cand_lh = glob.glob(os.path.join(root, '**', pat_lh), recursive=True)
        cand_rh = glob.glob(os.path.join(root, '**', pat_rh), recursive=True)
        if cand_lh and not lh: lh = cand_lh[0]
        if cand_rh and not rh: rh = cand_rh[0]
        if lh and rh:
            return {'lh': lh, 'rh': rh}

    # 3) Fallback: download the exact .annot files from CBIG Schaefer release
    cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'schaefer2018', mesh, 'label')
    os.makedirs(cache_dir, exist_ok=True)

    fname_lh = f"lh.Schaefer2018_{n_rois}Parcels_{yeo_networks}Networks_order.annot"
    fname_rh = f"rh.Schaefer2018_{n_rois}Parcels_{yeo_networks}Networks_order.annot"
    path_lh = os.path.join(cache_dir, fname_lh)
    path_rh = os.path.join(cache_dir, fname_rh)

    if not os.path.isfile(path_lh) or not os.path.isfile(path_rh):
        base_url = ("https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/"
                    "brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3")
        url_lh = f"{base_url}/{mesh}/label/{fname_lh}"
        url_rh = f"{base_url}/{mesh}/label/{fname_rh}"
        print(f"Downloading {url_lh}")
        urllib.request.urlretrieve(url_lh, path_lh)
        print(f"Downloading {url_rh}")
        urllib.request.urlretrieve(url_rh, path_rh)

    return {'lh': path_lh, 'rh': path_rh}


def schaefer_vector_to_vertices(parcel_vals, n_rois=100, yeo_networks=7, mesh='fsaverage5'):
    """
    Convert a Schaefer parcel vector (length n_rois) to per-vertex arrays on fsaverage.
    Returns: fsavg, lh_labels, rh_labels, lh_tex, rh_tex
    """
    fsavg = nl_datasets.fetch_surf_fsaverage(mesh=mesh)
    ann = _fetch_schaefer_annots(mesh=mesh, n_rois=n_rois, yeo_networks=yeo_networks)

    # Read .annot files: per-vertex label codes, color table, and names
    lh_labels, lh_ctab, lh_names = nib.freesurfer.read_annot(ann['lh'])
    rh_labels, rh_ctab, rh_names = nib.freesurfer.read_annot(ann['rh'])
    lh_names = _decode_fs_names(lh_names)
    rh_names = _decode_fs_names(rh_names)

    # -------- robust parcel filtering --------
    # Whitelist actual Schaefer parcels: names contain "Networks"
    def _only_parcels(ns):
        out = []
        for n in ns:
            s = n if isinstance(n, str) else (n.decode() if isinstance(n, (bytes, bytearray)) else str(n))
            if "networks" in s.lower():
                out.append(s)
        return out

    lh_names_clean = _only_parcels(lh_names)
    rh_names_clean = _only_parcels(rh_names)
    ordered_names = lh_names_clean + rh_names_clean

    if len(ordered_names) != len(parcel_vals):
        # helpful debug
        extras_l = [n for n in lh_names if "networks" not in (n.lower() if isinstance(n, str) else str(n).lower())][:3]
        extras_r = [n for n in rh_names if "networks" not in (n.lower() if isinstance(n, str) else str(n).lower())][:3]
        raise ValueError(
            f"Atlas length {len(ordered_names)} != vector length {len(parcel_vals)}. "
            f"Non-parcel examples: LH={extras_l} RH={extras_r}"
        )

    # Build robust lookup maps from labels to names. Some .annot files encode
    # per-vertex labels as the ctab last-column "code"; others may use row indices.
    # We support both by constructing two dicts and trying code first, then index.
    lh_codes = lh_ctab[:, -1]
    rh_codes = rh_ctab[:, -1]

    def _lookups(ctab_codes, names):
        by_code = {}
        by_index = {}
        for i, n in enumerate(names):
            s = n if isinstance(n, str) else (n.decode() if isinstance(n, (bytes, bytearray)) else str(n))
            if "networks" in s.lower():  # only map actual parcel rows
                by_code[ctab_codes[i]] = s
                by_index[i] = s
        return by_code, by_index

    by_code_l, by_index_l = _lookups(lh_codes, lh_names)
    by_code_r, by_index_r = _lookups(rh_codes, rh_names)

    name_to_val = {name: val for name, val in zip(ordered_names, parcel_vals)}

    # Paint: assign each vertex the value of its parcel name (skip non-parcels)
    def paint(labels, by_code, by_index):
        vals = np.full(labels.shape, np.nan, float)
        # Map unique labels in one pass for speed
        uniq = np.unique(labels)
        mapped_any = False
        for lab in uniq:
            name = by_code.get(lab)
            if name is None and 0 <= lab < len(by_index):
                name = by_index.get(int(lab))
            if name is None:
                continue  # medial wall / unknown / non-parcel
            mapped_any = True
            vals[labels == lab] = name_to_val[name]
        if not mapped_any:
            # Helpful warning during interactive use
            print("[warn] No atlas labels mapped to parcels — check .annot format and mesh.")
        return vals

    lh_tex = paint(lh_labels, by_code_l, by_index_l)
    rh_tex = paint(rh_labels, by_code_r, by_index_r)
    return fsavg, lh_labels, rh_labels, lh_tex, rh_tex

def save_side_by_side_iframe(left_html, right_html, out_html, title="Brain View", height="720px"):
    tpl = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <style>
    body {{ margin: 0; background: #fff; }}
    .wrap {{ display: flex; flex-direction: row; gap: 16px; padding: 16px; }}
    iframe {{ flex: 1 1 0; width: 50%; height: {height}; border: 1px solid #ddd; border-radius: 6px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <iframe src="{os.path.relpath(right_html, os.path.dirname(out_html))}"></iframe>
    <iframe src="{os.path.relpath(left_html, os.path.dirname(out_html))}"></iframe>
  </div>
</body>
</html>"""
    with open(out_html, "w") as f:
        f.write(tpl)
    print(f"[side-by-side] wrote {out_html}")
