#!/usr/bin/env python3
"""Verify fixed splits integrity."""

import json
from pathlib import Path

splits_file = Path("splits/split_v1.json")

with open(splits_file) as f:
    splits = json.load(f)

print("\nFIXED SPLITS VERIFICATION")
print("=" * 60)
print(f"Total samples in dataset: {splits['n_samples']}")
print(f"Train set:  {len(splits['train_indices'])} samples")
print(f"Val set:    {len(splits['val_indices'])} samples")
print(f"Test set:   {len(splits['test_indices'])} samples")

# Verify no overlap
train_set = set(splits['train_indices'])
val_set = set(splits['val_indices'])
test_set = set(splits['test_indices'])

overlap_tv = len(train_set & val_set)
overlap_tt = len(train_set & test_set)
overlap_vt = len(val_set & test_set)

print(f"\nLeakage Analysis:")
print(f"  overlap(train, val):  {overlap_tv}")
print(f"  overlap(train, test): {overlap_tt}")
print(f"  overlap(val, test):   {overlap_vt}")
print(f"  Total samples covered: {len(train_set | val_set | test_set)}/{splits['n_samples']}")

if overlap_tv == 0 and overlap_tt == 0 and overlap_vt == 0:
    print("\n✓✓✓ ZERO LEAKAGE - SPLITS ARE VALID AND REPRODUCIBLE ✓✓✓")
else:
    print("\n✗ LEAKAGE DETECTED - INVALID SPLITS")

# Verify split ratios
print(f"\nSplit Ratios:")
print(f"  Train: {len(splits['train_indices'])/splits['n_samples']:.1%}")
print(f"  Val:   {len(splits['val_indices'])/splits['n_samples']:.1%}")
print(f"  Test:  {len(splits['test_indices'])/splits['n_samples']:.1%}")

print(f"\nSplits file location: {splits_file}")
print("This file is reused across ALL 20 seeds to ensure")
print("the SAME test set is used for paired t-testing.\n")
