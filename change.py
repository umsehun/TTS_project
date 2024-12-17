import os

# 절대 경로의 기본 디렉토리 설정
base_dir = r"C:/Users/user/Desktop/furina_project/vits/dataset/wavs"

# 변환할 CSV 파일 경로
input_csv = r"C:/Users/user/Desktop/furina_project/vits/dataset/metadata.csv"
output_csv = r"C:/Users/user/Desktop/furina_project/vits/dataset/metadata_absolute.csv"

# 파일 처리
with open(input_csv, 'r', encoding='utf-8') as infile, open(output_csv, 'w', encoding='utf-8') as outfile:
    for line in infile:
        # 상대 경로 추출 (파일명은 첫 번째 파이프 `|` 이전)
        relative_path, *rest = line.split('|')
        
        # 절대 경로로 변환
        absolute_path = os.path.join(base_dir, os.path.basename(relative_path))
        # 다시 CSV 형식으로 저장
        outfile.write(f"{absolute_path}|{'|'.join(rest)}")
