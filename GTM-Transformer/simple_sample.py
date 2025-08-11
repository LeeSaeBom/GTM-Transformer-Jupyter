"""
간단한 데이터 샘플링 (pandas 없이)
"""
import os
import shutil
import random
from pathlib import Path

def simple_sample():
    # 경로 설정
    src_folder = Path('./dataset/')
    dst_folder = Path('./dataset_small/')
    
    # 대상 폴더 생성
    dst_folder.mkdir(exist_ok=True)
    (dst_folder / 'images').mkdir(exist_ok=True)
    
    print("Simple sampling started...")
    
    # train.csv sampling (every 10th line)
    print("Sampling train.csv...")
    with open(src_folder / 'train.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Header + every 10th line
    sampled_lines = [lines[0]]  # header
    for i in range(1, len(lines), 10):  # every 10th
        sampled_lines.append(lines[i])
    
    with open(dst_folder / 'train.csv', 'w', encoding='utf-8') as f:
        f.writelines(sampled_lines)
    
    print(f"  Created {len(sampled_lines)-1} train samples")
    
    # test.csv sampling (every 10th line)  
    print("Sampling test.csv...")
    with open(src_folder / 'test.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    sampled_lines = [lines[0]]  # header
    for i in range(1, len(lines), 10):  # every 10th
        sampled_lines.append(lines[i])
    
    with open(dst_folder / 'test.csv', 'w', encoding='utf-8') as f:
        f.writelines(sampled_lines)
    
    print(f"  Created {len(sampled_lines)-1} test samples")
    
    # Copy gtrends.csv
    print("Copying gtrends.csv...")
    shutil.copy2(src_folder / 'gtrends.csv', dst_folder / 'gtrends.csv')
    
    # Copy label files
    print("Copying label files...")
    for label_file in ['category_labels.pt', 'color_labels.pt', 'fabric_labels.pt']:
        shutil.copy2(src_folder / label_file, dst_folder / label_file)
    
    print("Sampling completed!")
    print(f"Saved to: {dst_folder}")

if __name__ == "__main__":
    simple_sample()