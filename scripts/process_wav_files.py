import os
import json
import csv

# 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
AUDIO_DIR = os.path.join(BASE_DIR, "dataset", "wavs")
JSON_FILES = [
    "C:\\Users\\user\\Desktop\\furina_project\\dataset\\dataset.json",
    "C:\\Users\\user\\Desktop\\furina_project\\dataset\\dataset_2.json",
    "C:\\Users\\user\\Desktop\\furina_project\\dataset\\dataset_3.json",
    "C:\\Users\\user\\Desktop\\furina_project\\dataset\\dataset_4.json",
    "C:\\Users\\user\\Desktop\\furina_project\\dataset\\dataset_5.json",
]
OUTPUT_CSV = os.path.join(BASE_DIR, "dataset", "metadata.csv")

# CSV 파일로 변환
def convert_json_to_csv(json_files, output_csv):
    with open(output_csv, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter="|")
        for json_file in json_files:
            if os.path.exists(json_file):
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for entry in data:
                        audio_filename = os.path.basename(entry["audio_filepath"])
                        text = entry["text"].strip()
                        writer.writerow([audio_filename, text])
                        print(f"Added: {audio_filename} | {text}")
            else:
                print(f"File not found: {json_file}")
    print(f"Metadata.csv created at: {output_csv}")

# 실행
convert_json_to_csv(JSON_FILES, OUTPUT_CSV)