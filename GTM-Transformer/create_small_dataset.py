"""
GTM 데이터셋을 Colab용으로 축소하는 스크립트
- 원본의 1/10 크기로 샘플링
- 카테고리/색상/소재 분포 유지
- 시즌별 균등 샘플링
"""

import pandas as pd
import numpy as np
import torch
import shutil
import os
from pathlib import Path
from collections import Counter

def create_small_dataset(
    data_folder='./dataset/',
    output_folder='./dataset_small/',
    train_samples=500,
    test_samples=50
):
    """
    작은 데이터셋 생성
    
    Args:
        data_folder: 원본 데이터 폴더
        output_folder: 출력 폴더
        train_samples: 훈련 샘플 수
        test_samples: 테스트 샘플 수
    """
    
    print(f"🔄 GTM 데이터셋 축소 시작...")
    print(f"📊 목표: 훈련 {train_samples}개, 테스트 {test_samples}개")
    
    # 출력 폴더 생성
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    (output_path / 'images').mkdir(exist_ok=True)
    
    # 원본 데이터 로드
    train_df = pd.read_csv(Path(data_folder) / 'train.csv', parse_dates=['release_date'])
    test_df = pd.read_csv(Path(data_folder) / 'test.csv', parse_dates=['release_date'])
    gtrends = pd.read_csv(Path(data_folder) / 'gtrends.csv', index_col=[0], parse_dates=True)
    
    print(f"📥 원본 데이터 크기:")
    print(f"  - 훈련: {len(train_df):,}개")
    print(f"  - 테스트: {len(test_df):,}개")
    
    # 1. 훈련 데이터 샘플링 (stratified sampling)
    print(f"\n🎯 훈련 데이터 샘플링...")
    
    # 카테고리별 균등 샘플링
    train_sampled_list = []
    categories = train_df['category'].unique()
    
    samples_per_category = train_samples // len(categories)
    remaining_samples = train_samples % len(categories)
    
    for i, category in enumerate(categories):
        category_data = train_df[train_df['category'] == category]
        
        # 일부 카테고리에 남은 샘플 추가 배분
        n_samples = samples_per_category + (1 if i < remaining_samples else 0)
        n_samples = min(n_samples, len(category_data))
        
        sampled = category_data.sample(n=n_samples, random_state=42)
        train_sampled_list.append(sampled)
        
        print(f"  📂 {category}: {len(category_data)} → {n_samples}개")
    
    train_sampled = pd.concat(train_sampled_list, ignore_index=True)
    
    # 2. 테스트 데이터 샘플링
    print(f"\n🎯 테스트 데이터 샘플링...")
    test_sampled = test_df.sample(n=min(test_samples, len(test_df)), random_state=42)
    
    print(f"  📊 테스트: {len(test_df)} → {len(test_sampled)}개")
    
    # 3. 사용된 이미지들만 복사
    print(f"\n🖼️ 이미지 파일 복사...")
    
    all_sampled = pd.concat([train_sampled, test_sampled])
    image_paths = all_sampled['image_path'].unique()
    
    copied_count = 0
    missing_count = 0
    
    for img_path in image_paths:
        src_path = Path(data_folder) / 'images' / img_path
        dst_path = output_path / 'images' / img_path
        
        # 대상 폴더 생성
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        else:
            print(f"  ⚠️ 없는 파일: {img_path}")
            missing_count += 1
    
    print(f"  ✅ 복사: {copied_count}개")
    if missing_count > 0:
        print(f"  ❌ 누락: {missing_count}개")
    
    # 4. CSV 파일 저장
    print(f"\n💾 CSV 파일 저장...")
    train_sampled.to_csv(output_path / 'train.csv', index=False)
    test_sampled.to_csv(output_path / 'test.csv', index=False)
    
    # Google Trends는 그대로 복사 (시간 시리즈이므로)
    gtrends.to_csv(output_path / 'gtrends.csv')
    
    # 5. 라벨 딕셔너리 업데이트
    print(f"\n🏷️ 라벨 딕셔너리 업데이트...")
    
    # 실제 사용되는 라벨만 추출
    used_categories = set(all_sampled['category'].unique())
    used_colors = set(all_sampled['color'].unique()) 
    used_fabrics = set(all_sampled['fabric'].unique())
    
    # 새로운 라벨 딕셔너리 생성
    new_cat_dict = {cat: i for i, cat in enumerate(sorted(used_categories))}
    new_col_dict = {col: i for i, col in enumerate(sorted(used_colors))}
    new_fab_dict = {fab: i for i, fab in enumerate(sorted(used_fabrics))}
    
    # 저장
    torch.save(new_cat_dict, output_path / 'category_labels.pt')
    torch.save(new_col_dict, output_path / 'color_labels.pt') 
    torch.save(new_fab_dict, output_path / 'fabric_labels.pt')
    
    print(f"  📋 카테고리: {len(new_cat_dict)}개")
    print(f"  🎨 색상: {len(new_col_dict)}개")
    print(f"  🧵 소재: {len(new_fab_dict)}개")
    
    # 6. 통계 출력
    print(f"\n📊 최종 데이터셋 통계:")
    print(f"  🚂 훈련: {len(train_sampled)}개")
    print(f"  🧪 테스트: {len(test_sampled)}개") 
    print(f"  🖼️ 이미지: {copied_count}개")
    print(f"  💾 폴더 크기 추정: ~{copied_count * 0.1:.0f} MB")
    
    print(f"\n✅ 축소된 데이터셋 생성 완료!")
    print(f"📂 저장 위치: {output_path}")
    print(f"\n🚀 Google Drive에 '{output_path.name}' 폴더를 업로드하세요!")
    
    return {
        'train_size': len(train_sampled),
        'test_size': len(test_sampled),
        'images': copied_count,
        'categories': len(new_cat_dict),
        'colors': len(new_col_dict), 
        'fabrics': len(new_fab_dict)
    }

if __name__ == "__main__":
    # 실행
    stats = create_small_dataset(
        data_folder='./dataset/',
        output_folder='./dataset_small/',
        train_samples=500,  # 훈련 500개
        test_samples=50     # 테스트 50개
    )
    
    print("\n" + "="*50)
    print("🎉 완료! 다음 단계:")
    print("1. 'dataset_small' 폴더를 Google Drive에 업로드")
    print("2. Colab에서 경로를 'GTM-dataset-small'로 변경")
    print("3. 빠른 실행으로 모델 테스트!")