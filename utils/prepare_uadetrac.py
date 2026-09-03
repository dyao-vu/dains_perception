#!/usr/bin/env python3
"""
Helper script to prepare UA-DETRAC test split by:
1. Identifying sequences with test annotations
2. Creating a test list file
3. Optionally organizing test images

Usage:
    python prepare_uadetrac_test.py --data_root dataset/UA-DETRAC
"""

import os
import argparse
import shutil
from pathlib import Path

def get_test_sequences_from_xmls(test_xml_folder: str):
    """Extract test sequence names from XML files."""
    sequences = []
    if os.path.exists(test_xml_folder):
        for f in os.listdir(test_xml_folder):
            if f.endswith('.xml'):
                # Remove .xml and potential _v3 suffix
                seq_name = f.replace('.xml', '').replace('_v3', '')
                if seq_name not in sequences:
                    sequences.append(seq_name)
    return sorted(sequences)

def create_test_list(sequences, output_file):
    """Create a text file with test sequence names."""
    with open(output_file, 'w') as f:
        for seq in sequences:
            f.write(f"{seq}\n")
    print(f"Created test list with {len(sequences)} sequences: {output_file}")

def organize_test_split(data_root, test_sequences, create_symlinks=True):
    """
    Optionally create a test folder structure similar to train.
    This creates: test/MVI_xxxxx/img1/imgXXXXXX.jpg
    """
    test_dir = os.path.join(data_root, 'test')
    images_source = os.path.join(data_root, 'DETRAC-Images')
    
    if not os.path.exists(images_source):
        print(f"Images folder not found: {images_source}")
        return
    
    os.makedirs(test_dir, exist_ok=True)
    
    created = 0
    for seq in test_sequences:
        src_seq = os.path.join(images_source, seq)
        if not os.path.exists(src_seq):
            print(f"  ✗ Missing images for {seq}")
            continue
        
        # Create test/MVI_xxxxx/img1 structure
        dst_seq = os.path.join(test_dir, seq, 'img1')
        
        if os.path.exists(dst_seq):
            print(f"  ⚠ Already exists: {seq}")
            created += 1
            continue
        
        os.makedirs(os.path.dirname(dst_seq), exist_ok=True)
        
        if create_symlinks:
            try:
                os.symlink(src_seq, dst_seq, target_is_directory=True)
                print(f"  ✓ Linked {seq}")
                created += 1
            except OSError:
                # Fallback to copying
                shutil.copytree(src_seq, dst_seq)
                print(f"  ✓ Copied {seq}")
                created += 1
        else:
            shutil.copytree(src_seq, dst_seq)
            print(f"  ✓ Copied {seq}")
            created += 1
    
    print(f"\nCreated test split with {created}/{len(test_sequences)} sequences in {test_dir}")
    return test_dir

def copy_test_annotations(data_root, test_sequences):
    """Copy test annotations to test/gt folder."""
    test_xml_folder = os.path.join(data_root, 'DETRAC-Annos' ,'DETRAC-Test-Annotations-XML')
    test_gt_folder = os.path.join(data_root, 'test', 'gt')
    
    if not os.path.exists(test_xml_folder):
        print(f"Test annotations not found: {test_xml_folder}")
        return
    
    os.makedirs(test_gt_folder, exist_ok=True)
    
    copied = 0
    for seq in test_sequences:
        xml_file = os.path.join(test_xml_folder, f"{seq}.xml")
        if not os.path.exists(xml_file):
            xml_file = os.path.join(test_xml_folder, f"{seq}_v3.xml")
        
        if os.path.exists(xml_file):
            dst_file = os.path.join(test_gt_folder, os.path.basename(xml_file))
            shutil.copy(xml_file, dst_file)
            copied += 1
    
    print(f"Copied {copied} annotation files to {test_gt_folder}")

def main():
    parser = argparse.ArgumentParser(description="Prepare UA-DETRAC test split")
    parser.add_argument('--data_root', required=True, help="Path to UA-DETRAC dataset root")
    parser.add_argument('--create_folder', action='store_true', 
                       help="Create test folder structure (like train/val)")
    parser.add_argument('--symlinks', action='store_true', 
                       help="Use symlinks instead of copying (saves space)")
    
    args = parser.parse_args()
    
    # Find test annotations
    test_xml_folder = os.path.join(args.data_root, 'DETRAC-Annos', 'DETRAC-Test-Annotations-XML')
    
    if not os.path.exists(test_xml_folder):
        print(f"✗ Test annotations folder not found: {test_xml_folder}")
        print("\nPlease download UA-DETRAC test annotations from:")
        print("https://detrac-db.rit.albany.edu/download")
        return
    
    # Get test sequences
    test_sequences = get_test_sequences_from_xmls(test_xml_folder)
    
    if not test_sequences:
        print("✗ No test sequences found")
        return
    
    print(f"\nFound {len(test_sequences)} test sequences")
    print(f"First 5: {test_sequences[:5]}")
    
    # Create test list file
    test_list_file = os.path.join(args.data_root, 'test_sequences.txt')
    create_test_list(test_sequences, test_list_file)
    
    # Optionally create test folder structure
    if args.create_folder:
        print("\n📁 Creating test folder structure...")
        organize_test_split(args.data_root, test_sequences, 
                          create_symlinks=args.symlinks)
        copy_test_annotations(args.data_root, test_sequences)
    
    print("\n✅ Done! You can now run evaluation with:")
    print(f"python eval_uadetrac.py --data_root {args.data_root} --split test")
    
    if not args.create_folder:
        print("\nNote: Test folder not created. Add --create_folder to create it.")

if __name__ == '__main__':
    main()