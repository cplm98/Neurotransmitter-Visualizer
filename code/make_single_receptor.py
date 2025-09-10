# -*- coding: utf-8 -*-
"""
Concatenate parcellated PET images into region x receptor matrix of densities.
"""

import numpy as np
from netneurotools import datasets, plotting
from matplotlib.colors import ListedColormap
from scipy.stats import zscore
from nilearn.datasets import fetch_atlas_schaefer_2018
import nibabel as nib
from nilearn import datasets as nl_datasets, plotting as nl_plotting
from netneurotools import datasets as nnt_datasets
import os

path = "/Users/connormoore/Documents/CS_Projects/Neurotransmitters/hansen_receptors/"

scale = 'scale100'

schaefer = fetch_atlas_schaefer_2018(n_rois=100)  # 100 parcels + 1 background label
labels = np.array(schaefer['labels'])

# Drop the first label ("Background")
if labels[0].lower().startswith('background'):
    labels = labels[1:]

nnodes = len(labels)  # == 100

# concatenate the receptors

receptors_csv = [path+'data/PET_parcellated/'+scale+'/5HT1a_way_hc36_savli.csv']
                #  path+'data/PET_parcellated/'+scale+'/5HT1b_p943_hc22_savli.csv',
                #  path+'data/PET_parcellated/'+scale+'/5HT1b_p943_hc65_gallezot.csv',
                #  path+'data/PET_parcellated/'+scale+'/5HT2a_cimbi_hc29_beliveau.csv',
                #  path+'data/PET_parcellated/'+scale+'/5HT4_sb20_hc59_beliveau.csv',
                #  path+'data/PET_parcellated/'+scale+'/5HT6_gsk_hc30_radhakrishnan.csv',
                #  path+'data/PET_parcellated/'+scale+'/5HTT_dasb_hc100_beliveau.csv',
                #  path+'data/PET_parcellated/'+scale+'/A4B2_flubatine_hc30_hillmer.csv',
                #  path+'data/PET_parcellated/'+scale+'/CB1_omar_hc77_normandin.csv',
                #  path+'data/PET_parcellated/'+scale+'/D1_SCH23390_hc13_kaller.csv',
                #  path+'data/PET_parcellated/'+scale+'/D2_flb457_hc37_smith.csv',
                #  path+'data/PET_parcellated/'+scale+'/D2_flb457_hc55_sandiego.csv',
                #  path+'data/PET_parcellated/'+scale+'/DAT_fpcit_hc174_dukart_spect.csv',
                #  path+'data/PET_parcellated/'+scale+'/GABAa-bz_flumazenil_hc16_norgaard.csv',
                #  path+'data/PET_parcellated/'+scale+'/H3_cban_hc8_gallezot.csv', 
                #  path+'data/PET_parcellated/'+scale+'/M1_lsn_hc24_naganawa.csv',
                #  path+'data/PET_parcellated/'+scale+'/mGluR5_abp_hc22_rosaneto.csv',
                #  path+'data/PET_parcellated/'+scale+'/mGluR5_abp_hc28_dubois.csv',
                #  path+'data/PET_parcellated/'+scale+'/mGluR5_abp_hc73_smart.csv',
                #  path+'data/PET_parcellated/'+scale+'/MU_carfentanil_hc204_kantonen.csv',
                #  path+'data/PET_parcellated/'+scale+'/NAT_MRB_hc77_ding.csv',
                #  path+'data/PET_parcellated/'+scale+'/NMDA_ge179_hc29_galovic.csv',
                # #  path+'data/PET_parcellated/'+scale+'/VAChT_feobv_hc3_spreng.csv',
                #  path+'data/PET_parcellated/'+scale+'/VAChT_feobv_hc4_tuominen.csv',
                #  path+'data/PET_parcellated/'+scale+'/VAChT_feobv_hc5_bedard_sum.csv',
                #  path+'data/PET_parcellated/'+scale+'/VAChT_feobv_hc18_aghourian_sum.csv']

# combine all the receptors (including repeats)
r = np.zeros([nnodes, len(receptors_csv)])
print("r shape: ", r.shape)
for i in range(len(receptors_csv)):
    r[:, i] = np.genfromtxt(receptors_csv[i], delimiter=',')

receptor_names = np.array(["5HT1a"]) 
                        #    "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                        #    "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                        #    "MOR", "NET", "NMDA", "VAChT"])
np.save(path+'data/receptor_names_pet.npy', receptor_names)

# make final region x receptor matrix

receptor_data = np.zeros([nnodes, len(receptor_names)])
print("receptor_data shape: ", receptor_data.shape)
receptor_data[:, 0] = r[:, 0]
print("receptor_data shape: ", receptor_data.shape)

# receptor_data[:, 2:9] = r[:, 3:10]
# receptor_data[:, 10:14] = r[:, 12:16]
# receptor_data[:, 15:18] = r[:, 19:22]

# weighted average of 5HT1B p943
# receptor_data[:, 1] = (zscore(r[:, 1])*22 + zscore(r[:, 2])*65) / (22+65)

# # weighted average of D2 flb457
# receptor_data[:, 9] = (zscore(r[:, 10])*37 + zscore(r[:, 11])*55) / (37+55)

# # weighted average of mGluR5 ABP688
# receptor_data[:, 14] = (zscore(r[:, 16])*22 + zscore(r[:, 17])*28 + zscore(r[:, 18])*73) / (22+28+73)

# # weighted average of VAChT FEOBV
# receptor_data[:, 18] = (zscore(r[:, 22])*3 + zscore(r[:, 23])*4 + zscore(r[:, 24]) + zscore(r[:, 25])) / \
#                        (3+4+5+18)

print(receptor_data.shape)

np.savetxt(path+'results/receptor_data_'+scale+'.csv', receptor_data, delimiter=',')


#-------------- Helper Functions ------------------#
from helper_functions import schaefer_vector_to_vertices, _fetch_schaefer_annots, _decode_fs_names



"""
plot receptor data
"""

# colourmaps
cmap = np.genfromtxt(path+'data/colourmap.csv', delimiter=',')
cmap_div = ListedColormap(cmap)

mesh = 'fsaverage6' # use fsaverage5 for fast, fsaverage6 for balance between detail and speed
image_type = "anatomical" # or inflated
n_rois = 100
yeo_networks = 7
outdir = "results/browser_demo/receptor_data"
os.makedirs(outdir, exist_ok=True)

# Load free surfer average mesh
fsavg = nl_datasets.fetch_surf_fsaverage(mesh=mesh)

#inflated vs pial
if image_type == 'anatomical': # pial is the anatomical mesh
    surf_left = fsavg.pial_left
    surf_right = fsavg.pial_right
elif image_type == "inflated": # inflated is more easy to visualize
    surf_left = fsavg.infl_left
    surf_right = fsavg.infl_right
else:
    print("invalid image type - please set correctly")
    exit()

# map receptor data

fsavg, _, _, lh_tex, rh_tex = schaefer_vector_to_vertices(receptor_data[:, 0], n_rois, yeo_networks, mesh)

# debug
print("Receptor data values:", receptor_data[:, 0])
print("lh_tex shape:", lh_tex.shape)
print("rh_tex shape:", rh_tex.shape)

# TODO: These are NaN, the data is not being mapped properly. I need to update the mapping funtion.
print("Left hemisphere texture values:", lh_tex)
print("Right hemisphere texture values:", rh_tex)


# Surface with parcel values mapped (surf_map) and sulcal background
# Note: nilearn.view_surf expects `surf_map`, not `texture`.
# Without the min/max the resulting graphically representation looks uniform in color.
vmin = np.nanpercentile(np.concatenate([lh_tex, rh_tex]), 2)
vmax = np.nanpercentile(np.concatenate([lh_tex, rh_tex]), 98)

view_left = nl_plotting.view_surf(
    surf_left,
    surf_map=lh_tex,
    bg_map=fsavg.sulc_left,
    cmap=cmap_div,
    colorbar=True,
    darkness=None,
    vmin=vmin, vmax=vmax,
)
view_right = nl_plotting.view_surf(
    surf_right,
    surf_map=rh_tex,
    bg_map=fsavg.sulc_right,
    cmap=cmap_div,
    colorbar=True,
    darkness=None,
    vmin=vmin, vmax=vmax,
)

view_left.save_as_html(os.path.join(outdir, f"03_values_left_{mesh}.html"))
view_right.save_as_html(os.path.join(outdir, f"03_values_right_{mesh}.html"))

# ---------------- helper to stitch left/right into one HTML ----------------

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


# ---------------- after each pair of saves, also stitch ----------------

# values
save_side_by_side_iframe(
    os.path.join(outdir, f"03_values_left_{mesh}.html"),
    os.path.join(outdir, f"03_values_right_{mesh}.html"),
    os.path.join(outdir, f"03_values_side_by_side_{mesh}.html"),
    title="Parcel values", height="720px"
)
