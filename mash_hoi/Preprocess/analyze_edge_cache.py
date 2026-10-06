"""
Analyze edge weight matrix cache to understand sparsity patterns.

Usage:
    python analyze_edge_cache.py
"""

import os
import h5py
import numpy as np
from collections import defaultdict


def analyze_cache(cache_path: str, num_samples: int = 100):
    """
    Analyze edge weight matrix cache.
    
    Args:
        cache_path: Path to the edge cache H5 file
        num_samples: Number of samples to analyze (0 = all)
    """
    print(f"Analyzing cache: {cache_path}")
    print("=" * 60)
    
    with h5py.File(cache_path, 'r') as hf:
        total_samples = len(hf.keys())
        print(f"Total samples in cache: {total_samples}")
        
        if num_samples <= 0 or num_samples > total_samples:
            num_samples = total_samples
        
        print(f"Analyzing {num_samples} samples...")
        print()
        
        # Statistics
        stats = defaultdict(list)
        
        for idx in range(num_samples):
            matrix = hf[str(idx)][:]  # [N, N]
            N = matrix.shape[0]
            
            # Basic stats
            nonzero_entries = (matrix > 0).sum()
            total_entries = matrix.size
            entry_sparsity = 1.0 - (nonzero_entries / total_entries)
            
            # Row stats (object patches)
            row_sums = matrix.sum(axis=1)  # [N]
            nonzero_rows = (row_sums > 0).sum()
            row_sparsity = 1.0 - (nonzero_rows / N)
            
            # Column stats (hand patches)
            col_sums = matrix.sum(axis=0)  # [N]
            nonzero_cols = (col_sums > 0).sum()
            col_sparsity = 1.0 - (nonzero_cols / N)
            
            # Value stats
            nonzero_values = matrix[matrix > 0]
            
            stats['nonzero_entries'].append(nonzero_entries)
            stats['entry_sparsity'].append(entry_sparsity)
            stats['nonzero_rows'].append(nonzero_rows)
            stats['row_sparsity'].append(row_sparsity)
            stats['nonzero_cols'].append(nonzero_cols)
            stats['col_sparsity'].append(col_sparsity)
            stats['max_value'].append(matrix.max())
            stats['mean_value'].append(matrix.mean())
            if len(nonzero_values) > 0:
                stats['mean_nonzero'].append(nonzero_values.mean())
            else:
                stats['mean_nonzero'].append(0)
            
            # Print first few samples in detail
            if idx < 5:
                print(f"Sample {idx}:")
                print(f"  Matrix shape: {matrix.shape}")
                print(f"  Non-zero entries: {nonzero_entries} / {total_entries} ({100*(1-entry_sparsity):.2f}%)")
                print(f"  Non-zero rows (object): {nonzero_rows} / {N} ({100*(1-row_sparsity):.2f}%)")
                print(f"  Non-zero cols (hand): {nonzero_cols} / {N} ({100*(1-col_sparsity):.2f}%)")
                print(f"  Value range: [{matrix.min():.4f}, {matrix.max():.4f}]")
                print(f"  Mean (all): {matrix.mean():.6f}, Mean (nonzero): {stats['mean_nonzero'][-1]:.4f}")
                print()
        
        # Summary statistics
        print("=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        
        print(f"\nEntry-level sparsity (fraction of zero entries):")
        print(f"  Mean: {np.mean(stats['entry_sparsity']):.4f}")
        print(f"  Std:  {np.std(stats['entry_sparsity']):.4f}")
        print(f"  Min:  {np.min(stats['entry_sparsity']):.4f}")
        print(f"  Max:  {np.max(stats['entry_sparsity']):.4f}")
        
        print(f"\nNon-zero entries per sample:")
        print(f"  Mean: {np.mean(stats['nonzero_entries']):.1f}")
        print(f"  Std:  {np.std(stats['nonzero_entries']):.1f}")
        print(f"  Min:  {np.min(stats['nonzero_entries'])}")
        print(f"  Max:  {np.max(stats['nonzero_entries'])}")
        
        print(f"\nRow sparsity (object patches, fraction of zero rows):")
        print(f"  Mean: {np.mean(stats['row_sparsity']):.4f}")
        print(f"  Std:  {np.std(stats['row_sparsity']):.4f}")
        
        print(f"\nNon-zero rows (object patches) per sample:")
        print(f"  Mean: {np.mean(stats['nonzero_rows']):.1f}")
        print(f"  Std:  {np.std(stats['nonzero_rows']):.1f}")
        print(f"  Min:  {np.min(stats['nonzero_rows'])}")
        print(f"  Max:  {np.max(stats['nonzero_rows'])}")
        
        print(f"\nColumn sparsity (hand patches, fraction of zero cols):")
        print(f"  Mean: {np.mean(stats['col_sparsity']):.4f}")
        print(f"  Std:  {np.std(stats['col_sparsity']):.4f}")
        
        print(f"\nNon-zero cols (hand patches) per sample:")
        print(f"  Mean: {np.mean(stats['nonzero_cols']):.1f}")
        print(f"  Std:  {np.std(stats['nonzero_cols']):.1f}")
        print(f"  Min:  {np.min(stats['nonzero_cols'])}")
        print(f"  Max:  {np.max(stats['nonzero_cols'])}")
        
        print(f"\nValue statistics:")
        print(f"  Max value (mean): {np.mean(stats['max_value']):.4f}")
        print(f"  Mean value (all): {np.mean(stats['mean_value']):.6f}")
        print(f"  Mean value (nonzero): {np.mean(stats['mean_nonzero']):.4f}")
        
        # Column vs Row comparison
        avg_nonzero_rows = np.mean(stats['nonzero_rows'])
        avg_nonzero_cols = np.mean(stats['nonzero_cols'])
        print(f"\n" + "=" * 60)
        print("RECOMMENDATION")
        print("=" * 60)
        if avg_nonzero_cols < avg_nonzero_rows:
            savings = (1 - avg_nonzero_cols / 256) * 100
            print(f"Columns (hand patches) are SPARSER:")
            print(f"  Avg non-zero cols: {avg_nonzero_cols:.1f} vs rows: {avg_nonzero_rows:.1f}")
            print(f"  Column classification saves ~{savings:.1f}% computation")
            print(f"  Current architecture (column classification) is OPTIMAL")
        else:
            savings = (1 - avg_nonzero_rows / 256) * 100
            print(f"Rows (object patches) are SPARSER:")
            print(f"  Avg non-zero rows: {avg_nonzero_rows:.1f} vs cols: {avg_nonzero_cols:.1f}")
            print(f"  Row classification would save ~{savings:.1f}% computation")
            print(f"  Consider switching back to row classification")


if __name__ == "__main__":
    # Cache paths
    cache_dir = "/mnt/d/yxz/DATASET/HOI-Dataset/Sphere-CMapDataset/.cache/edge_cache"
    
    # Analyze each hand type
    hand_types = ["shadow", "allegro", "barrett"]
    
    for hand in hand_types:
        cache_path = os.path.join(cache_dir, f"{hand}_edge_cache.h5")
        if os.path.exists(cache_path):
            print(f"\n{'#' * 60}")
            print(f"# {hand.upper()} HAND")
            print(f"{'#' * 60}\n")
            analyze_cache(cache_path, num_samples=1000)
        else:
            print(f"\nCache not found: {cache_path}")

