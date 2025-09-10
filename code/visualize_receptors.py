"""
Interactive receptor visualization with a dropdown selector.

For each parcellated PET CSV in data/PET_parcellated/<scale>, this script:
 - maps parcel values to fsaverage vertices
 - renders left/right hemisphere surfaces with a consistent color scale
 - writes a combined side-by-side HTML per dataset
 - builds an index HTML with a dropdown to switch between datasets

No Qt/Mayavi required; uses nilearn's plotly-based view_surf.
"""

import os
import sys
import glob
import numpy as np
from matplotlib.colors import ListedColormap
from nilearn import datasets as nl_datasets, plotting as nl_plotting

# Ensure local helpers are importable when running from repo root
sys.path.append(os.path.dirname(__file__))
from helper_functions import schaefer_vector_to_vertices, save_side_by_side_iframe


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def main():
    root = repo_root()
    scale = 'scale100'
    data_dir = os.path.join(root, 'data', 'PET_parcellated', scale)
    outdir = os.path.join(root, 'results', 'browser_demo', 'receptor_selector')
    os.makedirs(outdir, exist_ok=True)

    # Config
    mesh = 'fsaverage6'  # fsaverage5 is faster, fsaverage6 for balance
    image_type = 'anatomical'  # or 'inflated'
    n_rois = 100
    yeo_networks = 7
    # Visualization normalization
    normalize_mode = 'none'      # 'none' or 'zscore'
    center_zero = False          # if True, enforce symmetric vmin/vmax around 0 (diverging)
    reverse_cmap = False         # set True if the paper used reversed colors
    colormap_mode = 'diverging'  # 'diverging' or 'sequential' (paper sometimes used sequential)
    clamp_negatives = False      # if True and sequential, set negatives to 0 for display

    # Colormap
    cmap_path = os.path.join(root, 'data', 'colourmap.csv')
    cmap = np.genfromtxt(cmap_path, delimiter=',')
    cmap_div = ListedColormap(cmap)
    cmap_seq = ListedColormap(cmap[128:, :])  # upper half as sequential ramp
    if reverse_cmap:
        cmap_div = cmap_div.reversed()
        cmap_seq = cmap_seq.reversed()

    # Load fsaverage surfaces
    fsavg = nl_datasets.fetch_surf_fsaverage(mesh=mesh)
    if image_type == 'anatomical':
        surf_left = fsavg.pial_left
        surf_right = fsavg.pial_right
        bg_left = fsavg.sulc_left
        bg_right = fsavg.sulc_right
    elif image_type == 'inflated':
        surf_left = fsavg.infl_left
        surf_right = fsavg.infl_right
        bg_left = fsavg.sulc_left
        bg_right = fsavg.sulc_right
    else:
        raise ValueError("image_type must be 'anatomical' or 'inflated'")

    # Whitelist of receptor CSVs (mirrors make_single_receptor.py selection)
    allowed = [
        '5HT1a_way_hc36_savli.csv',
        '5HT1b_p943_hc22_savli.csv',
        '5HT1b_p943_hc65_gallezot.csv',
        '5HT1b_az_hc36_beliveau.csv',
        '5HT2a_cimbi_hc29_beliveau.csv',
        '5HT2a_alt_hc19_savli.csv',
        '5HT2a_mdl_hc3_talbot.csv',
        '5HT4_sb20_hc59_beliveau.csv',
        '5HT6_gsk_hc30_radhakrishnan.csv',
        '5HTT_dasb_hc100_beliveau.csv',
        '5HTT_dasb_hc30_savli.csv',
        'A4B2_flubatine_hc30_hillmer.csv',
        'CB1_FMPEPd2_hc22_laurikainen.csv',
        'CB1_omar_hc77_normandin.csv',
        'D1_SCH23390_hc13_kaller.csv',
        'D2_flb457_hc37_smith.csv',
        'D2_flb457_hc55_sandiego.csv',
        'D2_fallypride_hc49_jaworska.csv',
        'D2_raclopride_hc7_alakurtti.csv',
        'DAT_fepe2i_hc6_sasaki.csv',
        'DAT_fpcit_hc174_dukart_spect.csv',
        'GABAa-bz_flumazenil_hc16_norgaard.csv',
        'GABAa_flumazenil_hc6_dukart.csv',
        'H3_cban_hc8_gallezot.csv',
        'M1_lsn_hc24_naganawa.csv',
        'mGluR5_abp_hc22_rosaneto.csv',
        'mGluR5_abp_hc28_dubois.csv',
        'mGluR5_abp_hc73_smart.csv',
        'MU_carfentanil_hc204_kantonen.csv',
        'MU_carfentanil_hc39_turtonen.csv',
        'NAT_MRB_hc10_hesse.csv',
        'NAT_MRB_hc77_ding.csv',
        'NMDA_ge179_hc29_galovic.csv',
        # intentionally exclude VAChT_feobv_hc3_spreng.csv per original comment
        'VAChT_feobv_hc4_tuominen.csv',
        'VAChT_feobv_hc5_bedard_sum.csv',
        'VAChT_feobv_hc18_aghourian_sum.csv',
    ]
    csvs = []
    for name in allowed:
        full = os.path.join(data_dir, name)
        if os.path.isfile(full):
            csvs.append(full)
        else:
            print(f"[warn] Missing expected CSV: {name}")
    if not csvs:
        raise FileNotFoundError(f"None of the whitelisted CSVs were found in {data_dir}")

    # Prepare textures for all datasets
    entries = []  # list of dicts with: name, lh_tex, rh_tex, out_html

    for csv_path in csvs:
        try:
            parcel_vals = np.genfromtxt(csv_path, delimiter=',')
        except Exception as e:
            print(f"[skip] Failed to read {os.path.basename(csv_path)}: {e}")
            continue

        # Basic sanity: expect n_rois values
        if parcel_vals.ndim != 1 or parcel_vals.size != n_rois:
            print(f"[skip] {os.path.basename(csv_path)} has shape {parcel_vals.shape}, expected ({n_rois},)")
            continue

        # Optional normalization to better match paper conventions (often z-scored maps)
        if normalize_mode == 'zscore':
            mu = np.nanmean(parcel_vals)
            sd = np.nanstd(parcel_vals)
            if sd > 0 and np.isfinite(sd):
                parcel_vals = (parcel_vals - mu) / sd

        # Map to vertices
        _, _, _, lh_tex, rh_tex = schaefer_vector_to_vertices(
            parcel_vals, n_rois=n_rois, yeo_networks=yeo_networks, mesh=mesh
        )

        name = os.path.splitext(os.path.basename(csv_path))[0]
        entries.append({
            'name': name,
            'csv': csv_path,
            'lh_tex': lh_tex,
            'rh_tex': rh_tex,
        })
    # Optionally add weighted, z-scored combinations for specific receptor families
    # The weights reflect sample sizes from the original publications.
    # Only combinations whose source CSVs exist will be added.
    def _zscore(vec: np.ndarray) -> np.ndarray:
        v = vec.astype(float)
        m = np.nanmean(v)
        s = np.nanstd(v)
        if s == 0 or not np.isfinite(s):
            return v * 0.0
        return (v - m) / s

    # Map filename -> path for quick lookup
    available = {os.path.basename(p): p for p in csvs}

    combinations = {
        # 5HT1B p943 weighted
        '5HT1b_p943_weighted': [
            ('5HT1b_p943_hc22_savli.csv', 22),
            ('5HT1b_p943_hc65_gallezot.csv', 65),
        ],
        # D2 flb457 weighted
        'D2_flb457_weighted': [
            ('D2_flb457_hc37_smith.csv', 37),
            ('D2_flb457_hc55_sandiego.csv', 55),
        ],
        # mGluR5 ABP688 weighted
        'mGluR5_abp_weighted': [
            ('mGluR5_abp_hc22_rosaneto.csv', 22),
            ('mGluR5_abp_hc28_dubois.csv', 28),
            ('mGluR5_abp_hc73_smart.csv', 73),
        ],
        # VAChT FEOBV weighted
        'VAChT_feobv_weighted': [
            ('VAChT_feobv_hc3_spreng.csv', 3),
            ('VAChT_feobv_hc4_tuominen.csv', 4),
            ('VAChT_feobv_hc5_bedard_sum.csv', 5),
            ('VAChT_feobv_hc18_aghourian_sum.csv', 18),
        ],
        # Example: enable if desired
        # '5HTT_dasb_weighted': [
        #     ('5HTT_dasb_hc100_beliveau.csv', 100),
        #     ('5HTT_dasb_hc30_savli.csv', 30),
        # ],
    }

    for combo_name, items in combinations.items():
        paths_weights = [(available[f], w) for f, w in items if f in available]
        if len(paths_weights) < 2:
            # Skip combos without at least 2 available components
            continue
        zs = []
        ws = []
        for pth, w in paths_weights:
            vals = np.genfromtxt(pth, delimiter=',')
            if vals.ndim != 1 or vals.size != n_rois:
                continue
            zs.append(_zscore(vals))
            ws.append(float(w))
        if not zs:
            continue
        zs = np.stack(zs, axis=1)  # (n_rois, n_parts)
        ws = np.asarray(ws)
        wsum = ws.sum()
        if wsum == 0:
            continue
        weighted = (zs * ws).sum(axis=1) / wsum
        _, _, _, lh_tex, rh_tex = schaefer_vector_to_vertices(
            weighted, n_rois=n_rois, yeo_networks=yeo_networks, mesh=mesh
        )
        entries.append({
            'name': combo_name,
            'csv': None,
            'lh_tex': lh_tex,
            'rh_tex': rh_tex,
        })

    if not entries:
        raise RuntimeError("No valid receptor datasets were processed.")

    # Render per-dataset left/right HTML and a combined side-by-side
    for e in entries:
        name = e['name']
        lh_tex = e['lh_tex']
        rh_tex = e['rh_tex']

        # Per-dataset robust scaling improves visibility for low-dynamic-range maps
        cat = np.concatenate([lh_tex.ravel(), rh_tex.ravel()])
        cat = cat[~np.isnan(cat)]
        if cat.size == 0:
            vmin = vmax = 0.0
        else:
            if center_zero:
                m = np.percentile(np.abs(cat), 98)
                vmin, vmax = -m, m
            else:
                vmin = np.percentile(cat, 2)
                vmax = np.percentile(cat, 98)

        # Choose colormap
        use_cmap = cmap_div if colormap_mode == 'diverging' else cmap_seq

        # Optional clamping for sequential display
        if colormap_mode == 'sequential' and clamp_negatives:
            lh_plot = np.where(np.isnan(lh_tex), np.nan, np.maximum(lh_tex, 0))
            rh_plot = np.where(np.isnan(rh_tex), np.nan, np.maximum(rh_tex, 0))
        else:
            lh_plot, rh_plot = lh_tex, rh_tex

        view_left = nl_plotting.view_surf(
            surf_left,
            surf_map=lh_plot,
            bg_map=bg_left,
            cmap=use_cmap,
            colorbar=True,
            darkness=None,
            vmin=vmin, vmax=vmax,
        )
        view_right = nl_plotting.view_surf(
            surf_right,
            surf_map=rh_plot,
            bg_map=bg_right,
            cmap=use_cmap,
            colorbar=True,
            darkness=None,
            vmin=vmin, vmax=vmax,
        )

        left_html = os.path.join(outdir, f"{name}_left_{mesh}.html")
        right_html = os.path.join(outdir, f"{name}_right_{mesh}.html")
        both_html = os.path.join(outdir, f"{name}_side_by_side_{mesh}.html")
        view_left.save_as_html(left_html)
        view_right.save_as_html(right_html)
        save_side_by_side_iframe(left_html, right_html, both_html,
                                 title=f"{name}", height="720px")
        e['out_html'] = both_html

    # Build index page with dropdown that switches the iframe
    index_html = os.path.join(outdir, f"index_{mesh}.html")
    options = []
    for e in entries:
        rel = os.path.relpath(e['out_html'], start=os.path.dirname(index_html))
        options.append((e['name'], rel))

    html = [
        "<!doctype html>",
        "<html>",
        "<head>",
        "  <meta charset=\"utf-8\"/>",
        "  <title>Receptor selector</title>",
        "  <style>",
        "    body { font-family: -apple-system, system-ui, sans-serif; margin: 16px; }",
        "    .row { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }",
        "    iframe { width: 100%; height: 780px; border: 1px solid #ddd; border-radius: 6px; }",
        "    select { font-size: 14px; padding: 6px 8px; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <div class=\"row\">",
        "    <label for=\"receptor\"><strong>Select receptor dataset:</strong></label>",
        "    <select id=\"receptor\"></select>",
        "  </div>",
        "  <iframe id=\"viewer\"></iframe>",
        "  <script>",
        "    const options = [",
    ]

    for i, (name, rel) in enumerate(options):
        comma = ',' if i < len(options) - 1 else ''
        html.append(f"      {{\"label\": '{name}', \"value\": '{rel}'}}{comma}")

    html += [
        "    ];",
        "    const sel = document.getElementById('receptor');",
        "    const iframe = document.getElementById('viewer');",
        "    options.forEach(o => {",
        "      const opt = document.createElement('option');",
        "      opt.value = o.value;",
        "      opt.textContent = o.label;",
        "      sel.appendChild(opt);",
        "    });",
        "    function update() { iframe.src = sel.value; }",
        "    sel.addEventListener('change', update);",
        "    if (options.length > 0) { sel.selectedIndex = 0; update(); }",
        "  </script>",
        "</body>",
        "</html>",
    ]

    with open(index_html, 'w') as f:
        f.write("\n".join(html))

    print(f"Wrote {index_html}")


if __name__ == '__main__':
    main()
