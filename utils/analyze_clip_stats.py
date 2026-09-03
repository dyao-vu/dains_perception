#!/usr/bin/env python3
import pandas as pd
import numpy as np

df = pd.read_csv('clip_embedding_analysis_referkitti.csv')

print("="*60)
print("CLIP Similarity Statistics (GT Referred Objects)")
print("="*60)
print(f"Total samples: {len(df)}")
print(f"Mean:   {df['similarity'].mean():.4f}")
print(f"Std:    {df['similarity'].std():.4f}")
print(f"Min:    {df['similarity'].min():.4f}")
print(f"Max:    {df['similarity'].max():.4f}")
print(f"Median: {df['similarity'].median():.4f}")
print(f"Range:  {df['similarity'].max() - df['similarity'].min():.4f}")

print(f"\nPercentiles:")
for p in [5, 10, 25, 50, 75, 90, 95]:
    print(f"  {p:2d}th: {df['similarity'].quantile(p/100):.4f}")

print(f"\n{'='*60}")
print("Drop Rates at Different Thresholds:")
print("="*60)
for t in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    dropped = (df['similarity'] < t).mean()
    count = (df['similarity'] < t).sum()
    print(f"  thresh={t:.2f}: {dropped*100:5.1f}% dropped ({count:4d}/{len(df)} crops)")

print(f"\n{'='*60}")
print("Per-Expression Statistics:")
print("="*60)
expr_stats = df.groupby(['seq', 'expr_id', 'text'])['similarity'].agg([
    'mean', 'std', 'min', 'max', 'count'
])
print(expr_stats.to_string())

print(f"\n{'='*60}")
print("Object Size Correlation:")
print("="*60)
df['area'] = df['width'] * df['height']
corr = df[['similarity', 'area', 'width', 'height']].corr()
print(corr)

# Recommendations
print(f"\n{'='*60}")
print("RECOMMENDATIONS:")
print("="*60)
thresh_95 = df['similarity'].quantile(0.05)
thresh_90 = df['similarity'].quantile(0.10)
current_drop = (df['similarity'] < 0.20).mean()

print(f"\n1. Current setting (text_sim_thresh=0.20) drops {current_drop*100:.1f}% of GT objects")
if current_drop > 0.15:
    print("   ⚠️  TOO AGGRESSIVE - you're losing too many valid objects!")
elif current_drop > 0.05:
    print("   ⚠️  MODERATE - some GT objects being dropped")
else:
    print("   ✓ Acceptable drop rate")

print(f"\n2. Suggested thresholds:")
print(f"   - Conservative (keep 95%): text_sim_thresh={thresh_95:.3f}")
print(f"   - Moderate (keep 90%):     text_sim_thresh={thresh_90:.3f}")
print(f"   - Disable gating:          text_sim_thresh=0.0")

narrow_band = df['similarity'].std() < 0.05
print(f"\n3. Band analysis:")
print(f"   - Std deviation: {df['similarity'].std():.4f}")
if narrow_band:
    print("   ⚠️  NARROW BAND detected!")
    print("   - CLIP similarities have low variance")
    print("   - Hard to distinguish referred vs non-referred objects")
    print("   - Text gating provides minimal benefit")
    print("   - RECOMMENDATION: Use CLIP for re-ID, not gating")
else:
    print("   ✓ Good variance - CLIP can be discriminative")
