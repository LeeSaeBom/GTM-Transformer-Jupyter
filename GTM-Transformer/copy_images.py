"""
축소된 데이터셋에 필요한 이미지들만 복사
"""
import os
import shutil
from pathlib import Path

def copy_needed_images():
    src_img_folder = Path('./dataset/images/')
    dst_img_folder = Path('./dataset_small/images/')
    
    # 필요한 이미지 경로들 수집
    needed_images = set()
    
    # train.csv에서 이미지 경로 추출
    print("Extracting image paths from train.csv...")
    with open('./dataset_small/train.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:  # skip header
            parts = line.strip().split(',')
            if len(parts) > 20:  # image_path is at index 20
                img_path = parts[20]
                needed_images.add(img_path)
    
    # test.csv에서 이미지 경로 추출
    print("Extracting image paths from test.csv...")
    with open('./dataset_small/test.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:  # skip header
            parts = line.strip().split(',')
            if len(parts) > 20:  # image_path is at index 20
                img_path = parts[20]
                needed_images.add(img_path)
    
    print(f"Found {len(needed_images)} unique images needed")
    
    # 이미지 복사
    copied_count = 0
    missing_count = 0
    
    for img_path in needed_images:
        src_path = src_img_folder / img_path
        dst_path = dst_img_folder / img_path
        
        # 대상 폴더 생성
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            copied_count += 1
            if copied_count % 50 == 0:
                print(f"  Copied {copied_count} images...")
        else:
            print(f"  Missing: {img_path}")
            missing_count += 1
    
    print(f"Copy completed!")
    print(f"  Copied: {copied_count}")
    print(f"  Missing: {missing_count}")
    
    # 폴더 구조 확인
    print(f"\nFolder structure:")
    for subdir in dst_img_folder.iterdir():
        if subdir.is_dir():
            img_count = len(list(subdir.glob('*.png'))) + len(list(subdir.glob('*.jpg')))
            print(f"  {subdir.name}/: {img_count} images")

if __name__ == "__main__":
    copy_needed_images()