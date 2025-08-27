# -*- coding: utf-8 -*-
"""
Concatenate parcellated PET images into region x receptor matrix of densities.
"""

import numpy as np
import os
from netneurotools import datasets, plotting
from matplotlib.colors import ListedColormap
from scipy.stats import zscore
from nilearn.datasets import fetch_atlas_schaefer_2018
import plotly.graph_objects as go
from nilearn.surface import load_surf_mesh, vol_to_surf
from nilearn.datasets import fetch_surf_fsaverage
import nibabel as nib

path = "/Users/connormoore/Documents/CS_Projects/Neurotransmitters/hansen_receptors/"

scale = 'scale100'

schaefer = fetch_atlas_schaefer_2018(n_rois=100)  # 100 parcels + 1 background label
labels = np.array(schaefer['labels'])

# Drop the first label ("Background")
if labels[0].lower().startswith('background'):
    labels = labels[1:]

nnodes = len(labels)  # == 100

# concatenate the receptors

receptors_csv = [path+'data/PET_parcellated/'+scale+'/5HT1a_way_hc36_savli.csv',
                 path+'data/PET_parcellated/'+scale+'/5HT1b_p943_hc22_savli.csv',
                 path+'data/PET_parcellated/'+scale+'/5HT1b_p943_hc65_gallezot.csv',
                 path+'data/PET_parcellated/'+scale+'/5HT2a_cimbi_hc29_beliveau.csv',
                 path+'data/PET_parcellated/'+scale+'/5HT4_sb20_hc59_beliveau.csv',
                 path+'data/PET_parcellated/'+scale+'/5HT6_gsk_hc30_radhakrishnan.csv',
                 path+'data/PET_parcellated/'+scale+'/5HTT_dasb_hc100_beliveau.csv',
                 path+'data/PET_parcellated/'+scale+'/A4B2_flubatine_hc30_hillmer.csv',
                 path+'data/PET_parcellated/'+scale+'/CB1_omar_hc77_normandin.csv',
                 path+'data/PET_parcellated/'+scale+'/D1_SCH23390_hc13_kaller.csv',
                 path+'data/PET_parcellated/'+scale+'/D2_flb457_hc37_smith.csv',
                 path+'data/PET_parcellated/'+scale+'/D2_flb457_hc55_sandiego.csv',
                 path+'data/PET_parcellated/'+scale+'/DAT_fpcit_hc174_dukart_spect.csv',
                 path+'data/PET_parcellated/'+scale+'/GABAa-bz_flumazenil_hc16_norgaard.csv',
                 path+'data/PET_parcellated/'+scale+'/H3_cban_hc8_gallezot.csv', 
                 path+'data/PET_parcellated/'+scale+'/M1_lsn_hc24_naganawa.csv',
                 path+'data/PET_parcellated/'+scale+'/mGluR5_abp_hc22_rosaneto.csv',
                 path+'data/PET_parcellated/'+scale+'/mGluR5_abp_hc28_dubois.csv',
                 path+'data/PET_parcellated/'+scale+'/mGluR5_abp_hc73_smart.csv',
                 path+'data/PET_parcellated/'+scale+'/MU_carfentanil_hc204_kantonen.csv',
                 path+'data/PET_parcellated/'+scale+'/NAT_MRB_hc77_ding.csv',
                 path+'data/PET_parcellated/'+scale+'/NMDA_ge179_hc29_galovic.csv',
                #  path+'data/PET_parcellated/'+scale+'/VAChT_feobv_hc3_spreng.csv',
                 path+'data/PET_parcellated/'+scale+'/VAChT_feobv_hc4_tuominen.csv',
                 path+'data/PET_parcellated/'+scale+'/VAChT_feobv_hc5_bedard_sum.csv',
                 path+'data/PET_parcellated/'+scale+'/VAChT_feobv_hc18_aghourian_sum.csv']

# combine all the receptors (including repeats)
print("len(nnodes): ", nnodes)
r = np.zeros([nnodes, len(receptors_csv)])
print("shape(r): ", r.shape)
for i in range(len(receptors_csv)):
    r[:, i] = np.genfromtxt(receptors_csv[i], delimiter=',')

receptor_names = np.array(["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                           "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                           "MOR", "NET", "NMDA", "VAChT"])
np.save(path+'data/receptor_names_pet.npy', receptor_names)

# make final region x receptor matrix

receptor_data = np.zeros([nnodes, len(receptor_names)])
receptor_data[:, 0] = r[:, 0]
receptor_data[:, 2:9] = r[:, 3:10]
receptor_data[:, 10:14] = r[:, 12:16]
receptor_data[:, 15:18] = r[:, 19:22]

# weighted average of 5HT1B p943
receptor_data[:, 1] = (zscore(r[:, 1])*22 + zscore(r[:, 2])*65) / (22+65)

# weighted average of D2 flb457
receptor_data[:, 9] = (zscore(r[:, 10])*37 + zscore(r[:, 11])*55) / (37+55)

# weighted average of mGluR5 ABP688
receptor_data[:, 14] = (zscore(r[:, 16])*22 + zscore(r[:, 17])*28 + zscore(r[:, 18])*73) / (22+28+73)

# weighted average of VAChT FEOBV - had to update indexes because of missing files
receptor_data[:, 18] = (zscore(r[:, 21])*3 + zscore(r[:, 22])*4 + zscore(r[:, 23]) + zscore(r[:, 24])) / \
                       (3+4+5+18)

np.savetxt(path+'results/receptor_data_'+scale+'.csv', receptor_data, delimiter=',')


"""
plot receptor data
"""

### TO DO ###
# Update color maps to use the colormaps csv

# colourmaps
cmap = np.genfromtxt(path+'data/colourmap.csv', delimiter=',')
cmap_div = ListedColormap(cmap)

# Ensure the output directory exists
output_dir = path + 'figures/schaefer100/'
os.makedirs(output_dir, exist_ok=True)

# Path to your whole-brain .nii file
brain_nii_path = path + 'data/PET_nifti_images/5HT1a_cumi_hc8_beliveau.nii'

# Load the whole-brain .nii file
brain_img = nib.load(brain_nii_path)

# Fetch fsaverage surface (or use your own surface file)
fsaverage = fetch_surf_fsaverage()

# Use the pial surface for both hemispheres
lh_surface = fsaverage['pial_left']
rh_surface = fsaverage['pial_right']

# Load the surface meshes for both hemispheres
lh_mesh = load_surf_mesh(lh_surface)  # Left hemisphere
rh_mesh = load_surf_mesh(rh_surface)  # Right hemisphere

# Extract vertex coordinates
lh_coords = lh_mesh[0]  # Left hemisphere vertex coordinates
rh_coords = rh_mesh[0]  # Right hemisphere vertex coordinates

# Combine vertex coordinates
vertices = np.vstack((lh_coords, rh_coords))

# Extract faces (triangles) directly from the loaded meshes
lh_faces = lh_mesh[1]  # Left hemisphere faces
rh_faces = rh_mesh[1] + len(lh_coords)  # Adjust indices for right hemisphere

# Combine faces
faces = np.vstack((lh_faces, rh_faces))

# Project the Schaefer parcellation NIfTI file onto the surface
annot_lh = vol_to_surf(schaefer.maps, lh_surface)  # Left hemisphere labels
annot_rh = vol_to_surf(schaefer.maps, rh_surface)  # Right hemisphere labels

# Combine left and right hemisphere labels
annot = np.concatenate((annot_lh, annot_rh))

# Define camera views
from plotly.subplots import make_subplots

# Define cross-sectional views with axis ranges for cuts
cross_sections = {
    "frontal": dict(xaxis=dict(range=[-50, 50]), yaxis=dict(range=[-50, 50]), zaxis=dict(range=[-50, 50])),
    "sagittal": dict(xaxis=dict(range=[-50, 50]), yaxis=dict(range=[-50, 50]), zaxis=dict(range=[-50, 50])),
    "transverse": dict(xaxis=dict(range=[-50, 50]), yaxis=dict(range=[-50, 50]), zaxis=dict(range=[-50, 50]))
}

# Loop through each receptor and create a combined visualization
for k in range(len(receptor_names)):
    # Map the receptor density data to the vertices based on the parcellation
    vertex_colors = np.zeros(len(vertices))  # Initialize vertex colors
    for region_idx in range(len(labels)):
        # Assign the receptor density for the current region to the corresponding vertices
        vertex_colors[annot == region_idx + 1] = receptor_data[region_idx, k]  # +1 because labels are 1-based

    # Create the Plotly Mesh3d object
    brain_mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        intensity=vertex_colors,  # Color the vertices based on receptor data
        colorscale='Viridis',  # Choose a colorscale
        colorbar=dict(title=f'{receptor_names[k]} Density'),
        showscale=True,
        opacity=1
    )

    # Create a subplot figure for all views
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Frontal View", "Sagittal View", "Transverse View"],
        specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]]
    )

    # Add the brain mesh for each view
    for i, (view_name, axis_ranges) in enumerate(cross_sections.items(), start=1):
        fig.add_trace(brain_mesh, row=1, col=i)
        fig.update_scenes(
            dict(
                xaxis_title="X Axis",
                yaxis_title="Y Axis",
                zaxis_title="Z Axis",
                aspectmode="data",
                xaxis=axis_ranges["xaxis"],
                yaxis=axis_ranges["yaxis"],
                zaxis=axis_ranges["zaxis"]
            ),
            row=1, col=i
        )

    # Update the layout for the entire figure
    fig.update_layout(
        title=f"{receptor_names[k]} - Cross-Sectional Views",
        height=800,  # Adjust height for better visualization
        width=1200   # Adjust width for better visualization
    )

    # Save the plot as an HTML file
    fig.write_html(path + f'figures/schaefer100/surface_receptor_{receptor_names[k]}_cross_sections.html')