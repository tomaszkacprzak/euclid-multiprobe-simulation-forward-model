#!/usr/bin/env python3

import argparse
import pandas as pd


from pathlib import Path
from tqdm import tqdm
import h5py
import numpy as np
import pyarrow.parquet as pq

from pathlib import Path

import h5py
import pyarrow.parquet as pq


def parquet_to_hdf5_columns(
    parquet_path: str,
    hdf5_path: str,
    compression: str | None = "lzf",
):
    """
    Convert a Parquet file to an HDF5 file, storing each Parquet column
    as a separate extendable HDF5 dataset.

    Steps:
    1. Open input Parquet file using pyarrow.
    2. Build and print NumPy dtype for each column.
    3. Create one extendable HDF5 dataset per column.
    4. Loop over row groups, load each column, append it to its HDF5 dataset.
    5. Print total rows stored and close files.
    """

    parquet_path = Path(parquet_path)
    hdf5_path = Path(hdf5_path)

    pf = pq.ParquetFile(parquet_path)

    column_dtypes = {}

    for field in pf.schema_arrow:
        column_name = field.name

        try:
            numpy_dtype = field.type.to_pandas_dtype()
        except NotImplementedError:
            raise TypeError(
                f"Column {column_name!r} has Arrow type {field.type}, "
                "which cannot be directly converted to a NumPy dtype."
            )

        column_dtypes[column_name] = numpy_dtype

        print(f"{column_name:>30s}: {str(field.type):>10s}")

    print("Column NumPy dtypes:")
    for column_name, dtype in column_dtypes.items():
        print(f"{column_name:>40s}: {str(dtype):>10s}")

    total_rows = 0

    print(f"Converting {parquet_path} -> {hdf5_path}")
    with h5py.File(hdf5_path, "w") as h5:
        datasets = {}

        for column_name, dtype in column_dtypes.items():
            datasets[column_name] = h5.create_dataset(
                name=column_name,
                shape=(0,),
                maxshape=(None,),
                dtype=dtype,
                chunks=True,
                compression=compression,
                shuffle=True,
            )


        for row_group_index in range(pf.num_row_groups):
            table = pf.read_row_group(row_group_index)
            num_rows = table.num_rows

            for column_name in table.column_names:
                column = table[column_name]

                values = column.to_pandas().to_numpy()

                dataset = datasets[column_name]

                old_size = dataset.shape[0]
                new_size = old_size + num_rows

                dataset.resize((new_size,))
                dataset[old_size:new_size] = values

            total_rows += num_rows
            print(f"row group {row_group_index:>10d}/{pf.num_row_groups:>10d} stored {num_rows:>10d} rows, total {total_rows:>10d} rows")

    print(f"Total rows stored: {total_rows}")


def check_hdf5(hdf5_path: str):

    with h5py.File(hdf5_path, "r") as h5:

        for dataset_name in h5.keys():

            dataset = np.array(h5[dataset_name])
            dataset_finite = dataset[np.isfinite(dataset)]
            print(f"---------------------------------- column: {dataset_name:>20s}")
            print(f"   dtype={str(dataset.dtype):<30s}      shape={len(dataset):<30d}  num_finite={len(dataset_finite):<30d}")
            if len(dataset_finite) > 0:
                print(f'     min={dataset_finite.min(): 30.6f}        max={dataset_finite.max(): 30.6f}')
                print(f'    mean={dataset_finite.mean(): 30.6f}        std={dataset_finite.std(): 30.6f}')
            else:
                print(f'     min=NaN                            max=NaN')
                print(f'    mean=NaN                            std=NaN')
            print(f'num_nans={np.isnan(dataset).sum():<30d}   num_infs={np.isinf(dataset).sum():<30d}')

def check_unique_ids(hdf5_path: str):

    dataset = 'object_id'

    with h5py.File(hdf5_path, "r") as h5:
        
        ids = np.array(h5[dataset])
        
        unique_ids = np.unique(ids)
        
        print(f"Number of unique IDs: {len(unique_ids)} out of {len(ids)} total")
      
        


def main():
    parser = argparse.ArgumentParser(
        description="Convert a Parquet file into an HDF5 file."
    )
    parser.add_argument("input_parquet", help="Input Parquet file")
    parser.add_argument("output_hdf5", help="Output HDF5 file")
    args = parser.parse_args()

    # parquet_to_hdf5_columns(
    #     parquet_path=args.input_parquet,
    #     hdf5_path=args.output_hdf5,
    #     compression="lzf",
    # )

    check_hdf5(args.output_hdf5)
    check_unique_ids(args.output_hdf5)


if __name__ == "__main__":
    main()