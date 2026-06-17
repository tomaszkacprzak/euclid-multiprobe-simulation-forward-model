import os, sys


def get_filename_data_vectors(out_dir, with_bary=False):
    if with_bary:
        file_name = "data_vectors_baryonified.h5"
    else:
        file_name = "data_vectors_nobaryons.h5"

    return os.path.join(out_dir, file_name)


def get_filename_data_patches(out_dir, with_bary=False):
    if with_bary:
        file_name = "data_patches_baryonified.h5"
    else:
        file_name = "data_patches_nobaryons.h5"

    return os.path.join(out_dir, file_name)


def get_filename_full_maps(grid_dir="", with_bary=False, version="1.1"):
    if with_bary:
        if version == "1":
            file_name = "projected_probes_maps_baryonified512.h5"
        elif version == "1.1":
            file_name = "projected_probes_maps_v11dmb.h5"
    else:
        if version == "1":
            file_name = "projected_probes_maps_nobaryons512.h5"
        elif version == "1.1":
            file_name = "projected_probes_maps_v11dmo.h5"

    return os.path.join(grid_dir, file_name)

def get_filename_examples_name(index, tag, simset, with_bary=False, return_pattern=False):

    if return_pattern:
        index = "????"
    else:
        index = f"{index:04d}"

    if with_bary:
        file_name = f"{tag}_{simset}_dmb_{index}"
    else:
        file_name = f"{tag}_{simset}_dmo_{index}"

    return file_name


def get_filename_webdataset(out_dir, index, tag, simset, with_bary=False, return_pattern=False):
    fname = get_filename_examples_name(index, tag, simset, with_bary, return_pattern)
    return os.path.join(out_dir, fname+'.tar')


def get_filename_tfrecords(out_dir, index, tag, simset, with_bary=False, return_pattern=False):
    fname = get_filename_examples_name(index, tag, simset, with_bary, return_pattern)
    return os.path.join(out_dir, fname+'.tfrecord')


def get_filename_z_distribution(conf, data_dir, galaxy_sample_label, i_bin):

    if conf["survey"]["name"] == "DESy3":
        fname = os.path.join(data_dir, f"desy3_nz_{galaxy_sample_label.upper()}_bin{i_bin}.txt")
    
    elif conf["survey"]["name"] == "EuclidDR1F":
        fname = os.path.join(data_dir, f"RR2_v2_Nz_GC_C2020_sel_pv.{galaxy_sample_label.upper()}.{i_bin}.smoothed.txt")
    
    else:
        raise ValueError(f"Survey {conf['survey']['name']} not supported")
    
    return fname

