#!/usr/bin/python
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "h5py",
#     "numpy",
#     "polars",
#     "scipy",
# ]
# ///

#-------------------------------------------------------
# Score test to evaluate the significance of 
# new annotations

# Note: this function requires a suite of three output 
# files from graphREML estimation. 
# 1) pre-computed SNP-level of gradient scores and hessian (*_perSNPh2.txt)
# this file is saved when specifying nullFit as True in run_graphREML
# 2) jackknifed estimates (*_paramsJK.txt)
# 3) estimates table (*_est.txt)


#-------------------------------------------------------
import os, sys, re
import logging, time, traceback
import argparse
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
import polars as pl
import numpy as np
from scipy import stats
import h5py
import importlib.util

# To import load_annotations without triggering graphld/__init__.py,
# we load the io.py module directly from its file path.
io_path = Path(__file__).resolve().parent / "io.py"
spec = importlib.util.spec_from_file_location("graphld.io", io_path)
graphld_io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graphld_io)
load_annotations = graphld_io.load_annotations

def _load_variant_data(hdf5_path: str) -> dict:
    """
    Load variant statistics from HDF5 file format.
    
    Args:
        hdf5_path: Path to the HDF5 file containing variant statistics
        
    Returns:
        Polars DataFrame containing variant statistics
    """
    
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # Check for required groups
        if 'variants' not in f or 'traits' not in f:
            raise ValueError(f"HDF5 file {hdf5_path} does not have the expected structure with 'variants' and 'traits' groups")
        
        # Get variant information
        variants_group = f['variants']
        
        # Create a dictionary to store data
        data = {
            'POS': variants_group['POS'][:],
            'annotations': variants_group['annotations'][:,:],
        }
        
        # Handle string data - convert bytes to strings
        rsids = variants_group['RSID'][:]
        data['RSID'] = [s[0].decode('utf-8') if isinstance(s[0], bytes) else s[0] for s in rsids]
        
    return data
    

def _load_trait_data(hdf5_path: str, trait_name: str) -> dict:
    """
    Load trait data from HDF5 file format.
    """
    
    with h5py.File(hdf5_path, 'r') as f:
        trait_group = f[f'traits/{trait_name}']
        keys = ['parameters', 'jackknife_parameters', 'jackknife_blocks', 'gradient', 'hessian']
        data = {key: trait_group[key][:] for key in keys}
    
    return data

def _get_block_boundaries(blocks: np.ndarray) -> np.ndarray:
    num_blocks = len(np.unique(blocks))
    temp = np.where(np.diff(blocks) != 0)[0]
    if len(temp) != num_blocks - 1:
        raise ValueError(f"Jackknife block indices in merged dataframe should be contiguous ",
            f"but there were {len(temp)} differences and {num_blocks} unique blocks")
    block_boundaries = np.zeros(num_blocks + 1, dtype=int)
    block_boundaries[1:-1] = temp
    block_boundaries[-1] = len(blocks)
    return block_boundaries

def _add_random_annotations(random_annotation_params: Optional[str], df_annot: pl.DataFrame) -> Tuple[pl.DataFrame, List[str]]:
    """Adds random annotations based on provided parameters and returns the modified DataFrame and added column names."""
    random_annot_names = []
    if random_annotation_params:
        random_probs = [float(p) for p in random_annotation_params.split(',')]
        num_variants = len(df_annot)
        for i, p in enumerate(random_probs):
            if p < 0 or p > 1:
                raise ValueError(f"Random annotation probability must be between 0 and 1, got {p}")
            col_name = f"random_{i}"
            random_annot_names.append(col_name)
            random_column = np.random.binomial(1, p, size=num_variants)
            df_annot = df_annot.with_columns(pl.Series(name=col_name, values=random_column))
            logging.info(f"Added random annotation '{col_name}' with p={p}")
    return df_annot, random_annot_names

def _project_out(y: np.ndarray, x: np.ndarray):
    """Projects out x from y in place."""
    beta = np.linalg.solve(x.T @ x, x.T @ y )
    y -= x @ beta

def run_score_test(
    df_snp: pl.DataFrame, 
    df_annot: pl.DataFrame, 
    params: np.ndarray, 
    jk_params: np.ndarray, 
    annot_test_list: List[str],
) -> np.ndarray:
    """
    Run score test for hypothesis testing of new annotation or functional category.
    
    Args:
        df_snp: DataFrame containing variant-level statistics
        df_annot: DataFrame containing annotations
        params: Parameter values from the null model
        jk_params: Jackknife estimates for the null model
        annot_test_list: List of annotations to test.
        adjust_uncertain_fit: Whether to adjust for uncertainty in the model fit when computing 
            jackknife derivatives. Recommended to be False for exploratory analyses.
        
    Returns:
        Z score for enrichment of each annotation
    """
    
    # Validate inputs
    required_snp_cols = ['RSID', 'POS', 'gradient', 'hessian', 'jackknife_blocks']
    if not all(col in df_snp.columns for col in required_snp_cols):
        raise ValueError(f"Missing required columns in df_snp: {', '.join(required_snp_cols)}")
    if not all(col in df_annot.columns for col in annot_test_list):
        raise ValueError(f"Missing required columns in df_annot: {', '.join(annot_test_list)}")
    
    # Select only the columns we need from annotations
    df_annot = df_annot.select(['SNP'] + annot_test_list)
    
    # Merge variant statistics with annotations
    df_merged = df_snp.join(df_annot, left_on='RSID', right_on='SNP', how='inner', maintain_order='left')
    logging.info(f"Columns in merged dataframe: {df_merged.columns}")
    logging.info(f"Number of rows in merged dataframe: {df_merged.shape[0]}")
    block_boundaries = _get_block_boundaries(df_merged['jackknife_blocks'].to_numpy())
    noBlocks = len(block_boundaries) - 1
    logging.info(f"Number of blocks to jackknife from: {noBlocks}")
    variant_grad = df_merged['gradient'].to_numpy().astype(np.float64)
    test_annot = df_merged[annot_test_list].to_numpy().astype(np.float64)
    model_annot = df_merged['annotations'].to_numpy().astype(np.float64)
    variant_hess = df_merged['hessian'].to_numpy().astype(np.float64)

    _project_out(test_annot, model_annot)

    # Compute single-block derivatives
    U_block = [] # Derivative of the log-likelihood for each test annotation
    J_block = [] # Derivative of U_block w.r.t. each model parameter
    for i in range(noBlocks):
        block = range(block_boundaries[i], block_boundaries[i+1])

        Ui = variant_grad[block].reshape(1,-1) @ test_annot[block, :]
        U_block.append(Ui)

        Ji = model_annot[block, :].T @ (variant_hess[block].reshape(-1,1) * test_annot[block, :])
        J_block.append(Ji)

    U_total = sum(U_block)
    J_total = sum(J_block)

    # Compute leave-one-out derivatives
    U_jackknife = np.zeros((noBlocks, U_total.shape[1]))
    for i in range(noBlocks):
        block = range(block_boundaries[i], block_boundaries[i+1])

        # deduct the score contributed from one LD block
        U_jackknife[i, :] = U_total - U_block[i].ravel()

        # correct for the change in parameters when leaving out this block
        Ji = J_total - J_block[i]
        U_jackknife[i, :] += ((jk_params[i,:] - params).reshape(1,-1) @ Ji).ravel()
    
    # Compute jackknife Z scores
    jack_mean = U_total.ravel()
    jack_se = np.std(U_jackknife, axis=0) * np.sqrt(noBlocks-1)
    return jack_mean / jack_se


def main():
    parser = argparse.ArgumentParser(description='Run score test for annotation enrichment.')
    
    parser.add_argument("variant_stats_hdf5", help="Path to the HDF5 file containing precomputed derivatives.")
    parser.add_argument("output_fp", help="Output file path prefix (e.g., 'results/my_test'). '.txt' and '.log' will be appended.")
    parser.add_argument("-a", "--annotations_dir", required=True, help="Directory containing annotation files to test (e.g., .annot, .bed). Required.")
    parser.add_argument("-n", "--name", help="Specific trait name to process from HDF5 file. If omitted, all traits are processed.")
    parser.add_argument("--annotations", help="Optional comma-separated list of specific annotation names (columns) to test.")
    parser.add_argument("--add-random", dest="add_random", help="Comma-separated list of probabilities (0-1) to generate random binary annotations for testing (e.g., '0.1,0.01').")
    parser.add_argument("--stream-stdout", dest="stream_stdout", action='store_true', help='Stream log information on console in addition to writing to log file.')
    parser.add_argument("--seed", type=int, default=None, help='Seed for generating random annotations.')
    
    args = parser.parse_args()
    
    #-----------------------------
    # Set up logging
    #-----------------------------
    log_fp = f"{args.output_fp}_scoretest.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=([logging.FileHandler(log_fp), logging.StreamHandler(sys.stdout)] 
                  if args.stream_stdout 
                  else [logging.FileHandler(log_fp)])
    )
    start_time = time.time()

    # Load variant statistics from HDF5 file
    logging.info(f"Loading variant statistics from HDF5 file: {args.variant_stats_hdf5}")
    variant_data = _load_variant_data(args.variant_stats_hdf5)
        
    # Load annotations from directory
    logging.info(f"Loading annotations from directory: {args.annotations_dir}")
    df_annot = load_annotations(args.annotations_dir, add_positions=False)
    
    # Add random annotations if requested
    np.random.seed(args.seed)
    df_annot, random_annot_names = _add_random_annotations(args.add_random, df_annot)

    # Determine the final list of annotations to test
    final_annot_test_list = args.annotations.split(',') if args.annotations else []
    final_annot_test_list.extend(random_annot_names)
    
    # If no annotations were specified or generated, use defaults from df_annot
    if not final_annot_test_list:
        logging.info("No specific or random annotations provided. Detecting annotations from annotation file...")
        excluded_cols = {'SNP', 'CM', 'BP', 'CHR', 'POS', 'A1', 'A2', 'RSID'} 
        potential_annots = [col for col in df_annot.columns if col not in excluded_cols]
        final_annot_test_list = potential_annots
    logging.info(f"Using annotations: {', '.join(final_annot_test_list)}")

    # Determine traits to process (succinctly)
    trait_names = [args.name] if args.name else list(h5py.File(args.variant_stats_hdf5, 'r')['traits'].keys())

    # Run the score test
    results_dict = {'annotation' : final_annot_test_list}
    for trait in trait_names:
        # Load parameter estimates and jackknife estimates from HDF5 file
        trait_data = _load_trait_data(args.variant_stats_hdf5, trait_name=trait)
        variant_data['gradient'] = trait_data['gradient']
        variant_data['hessian'] = trait_data['hessian']
        variant_data['jackknife_blocks'] = trait_data['jackknife_blocks']
        
        score_test_Z_scores = run_score_test(
            df_snp=pl.DataFrame(variant_data),
            df_annot=df_annot,
            params=trait_data['parameters'],
            jk_params=trait_data['jackknife_parameters'],
            annot_test_list=final_annot_test_list,
        )
        results_dict[trait] = score_test_Z_scores

    results_df = pl.DataFrame(results_dict)
    print(results_df)    
    
    # Write results to file
    results_df.write_csv(args.output_fp + ".txt", separator='\t')
    
    logging.info(f'Total time elapsed: {time.time()-start_time}')
    
if __name__ == '__main__':
    main()