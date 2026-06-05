#!/usr/bin/env python3
import os
import argparse
import h5py
import numpy as np


# https://euclid.roe.ac.uk/projects/dr1-kp-jc-2/wiki/Selection_history#Position-sample
# TR1 v2.0



bins_clustering= []
bins_clustering.append([0.2, 1.0]) # bin 0 is total
bins_clustering.append([0.2, 0.34])
bins_clustering.append([0.34, 0.49])
bins_clustering.append([0.49, 0.59])
bins_clustering.append([0.59, 0.71])
bins_clustering.append([0.71, 0.83])
bins_clustering.append([0.83, 1.0])


bins_lensing= []
bins_lensing.append([0.2, 0.48])
bins_lensing.append([0.48, 0.67])
bins_lensing.append([0.67, 0.86])
bins_lensing.append([0.86, 1.07])
bins_lensing.append([1.07, 1.42])
bins_lensing.append([1.42, 2.5])


def create_dataset(h5_output, name, data):

    print(f"Creating dataset {name} with shape {data.shape} and dtype {data.dtype}")

    if name in h5_output:
        del h5_output[name]
    
    h5_output.create_dataset(name=name, data=data)
    


def main():
    parser = argparse.ArgumentParser(
        description="Convert a Parquet file into an HDF5 file."
    )
    parser.add_argument("input_lensing_hdf5", help="Input Lensing HDF5 file")
    parser.add_argument("input_clustering_hdf5", help="Input Clustering HDF5 file")
    parser.add_argument("output_hdf5", help="Output HDF5 file")
    args = parser.parse_args()



    # clustering bins
    print(f"------------- Processing clustering bins")

    z_bins_clustering_edges = np.arange(0.2, 1.0+0.001, 0.001)
    z_bins_clustering_centers = (z_bins_clustering_edges[:-1] + z_bins_clustering_edges[1:]) / 2
    nz_file_path = os.path.join(os.path.dirname(args.output_hdf5), "tr1v2.0_GC_nz_bin{}.txt")

    with h5py.File(args.input_clustering_hdf5, "r") as h5_clustering:
        ra = np.array(h5_clustering['right_ascension'])
        dec = np.array(h5_clustering['declination'])
        object_id = np.array(h5_clustering['object_id'])
        z = np.array(h5_clustering['phz_mode_1'])

        mask_valid = (np.isfinite(z)) & (np.isfinite(ra)) & (np.isfinite(dec)) & (np.isfinite(object_id)) 

        for bin_id, bin_ in enumerate(bins_clustering):
            mask = (z > bin_[0]) & (z < bin_[1]) & mask_valid
            object_id_bin = object_id[mask]
            z_bin = z[mask]
            ra_bin = ra[mask]
            dec_bin = dec[mask]

            with h5py.File(args.output_hdf5, "w") as h5_output:
                create_dataset(h5_output, f"GC/bin{bin_id}/id", object_id_bin)
                create_dataset(h5_output, f"GC/bin{bin_id}/z", z_bin)
                create_dataset(h5_output, f"GC/bin{bin_id}/ra", ra_bin)
                create_dataset(h5_output, f"GC/bin{bin_id}/dec", dec_bin)

                print(f"Clustering bin {bin_id} {bin_[0]} - {bin_[1]}: {len(object_id_bin)} galaxies")

                nz_bins_clustering = np.histogram(z_bin, bins=z_bins_clustering_edges)[0].astype(np.float32)
                nz_bins_clustering = nz_bins_clustering/np.sum(nz_bins_clustering)
                create_dataset(h5_output, f"GC/bin{bin_id}/binned_nz", nz_bins_clustering)
                create_dataset(h5_output, f"GC/bin{bin_id}/binned_z_edges", z_bins_clustering_edges.astype(np.float32))
                create_dataset(h5_output, f"GC/bin{bin_id}/binned_z_centers", z_bins_clustering_centers.astype(np.float32))

            znz = np.column_stack((z_bins_clustering_centers, nz_bins_clustering))
            fname_nz_bin = nz_file_path.format(bin_id)
            np.savetxt(fname_nz_bin, znz)
            print(f'Saved nz file for clustering bin {bin_id} to {fname_nz_bin}')


            

    # lensing bins
    print(f"------------- Processing lensing bins")

    z_bins_lensing_edges = np.arange(0.2, 2.5+0.001, 0.001)
    z_bins_lensing_centers = (z_bins_lensing_edges[:-1] + z_bins_lensing_edges[1:]) / 2
    nz_file_path = os.path.join(os.path.dirname(args.output_hdf5), "tr1v2.0_nz_WL_bin{}.txt")

    with h5py.File(args.input_lensing_hdf5, "r") as h5_lensing:

        e1 = np.array(h5_lensing['she_lensmc_e1'])
        e2 = np.array(h5_lensing['she_lensmc_e2'])
        e_var = np.array(h5_lensing['she_lensmc_e_var'])
        ra = np.array(h5_lensing['right_ascension'])
        dec = np.array(h5_lensing['declination'])
        object_id = np.array(h5_lensing['object_id'])
        z = np.array(h5_lensing['phz_mode_1'])

        mask_valid = (np.isfinite(z)) & (np.isfinite(ra)) & (np.isfinite(dec)) & (np.isfinite(object_id)) & (np.isfinite(e1)) & (np.isfinite(e2)) & (np.isfinite(e_var))

        for bin_id, bin_ in enumerate(bins_lensing):
            mask = (z > bin_[0]) & (z < bin_[1]) & mask_valid
            e1_bin = e1[mask]
            e2_bin = e2[mask]
            e_var_bin = e_var[mask]
            object_id_bin = object_id[mask]
            z_bin = z[mask]
            
            with h5py.File(args.output_hdf5, "w") as h5_output:
                create_dataset(h5_output, f"WL/bin{bin_id}/e1", e1_bin)
                create_dataset(h5_output, f"WL/bin{bin_id}/e2", e2_bin)
                create_dataset(h5_output, f"WL/bin{bin_id}/e_var", e_var_bin)
                create_dataset(h5_output, f"WL/bin{bin_id}/id", object_id_bin)   
                create_dataset(h5_output, f"WL/bin{bin_id}/z", z_bin)

                print(f"Lensing bin {bin_id} {bin_[0]} - {bin_[1]}: {len(e1_bin)} galaxies")

                nz_bins_lensing = np.histogram(z_bin, bins=z_bins_lensing_edges)[0].astype(np.float32)
                nz_bins_lensing = nz_bins_lensing/np.sum(nz_bins_lensing)
                create_dataset(h5_output, f"WL/bin{bin_id}/binned_nz", nz_bins_lensing)
                create_dataset(h5_output, f"WL/bin{bin_id}/binned_z_edges", z_bins_lensing_edges)
                create_dataset(h5_output, f"WL/bin{bin_id}/binned_z_centers", z_bins_lensing_centers)

            znz = np.column_stack((z_bins_lensing_centers, nz_bins_lensing))
            fname_nz_bin = nz_file_path.format(bin_id)
            np.savetxt(fname_nz_bin, znz)
            print(f'Saved nz file for lensing bin {bin_id} to {fname_nz_bin}')

    


if __name__ == "__main__":
    main()