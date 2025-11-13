# app/init_data.py
import os
import urllib.request
import zipfile
import shutil
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "app/data"
WEIGHTS_DIR = "app/models/"

DATA_ZIP_PATH = "app/data/anime-datasets.zip"
WEIGHTS_ZIP_PATH = "app/models/weights.zip"

DATA_BUNDLE_URL = os.getenv("DATA_BUNDLE_URL")
WEIGHTS_BUNDLE_URL = os.getenv("WEIGHTS_BUNDLE_URL")


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def download_if_missing(url: str, zip_path: str):
    """파일이 없으면 ZIP 다운로드"""
    if not url:
        print(f"[init_data] URL 없음: {url}")
        return False

    if os.path.exists(zip_path):
        print(f"[init_data] ZIP 이미 존재: {zip_path}")
        return False

    print(f"[init_data] 다운로드 시작: {url}")

    # 스트리밍 다운로드
    with urllib.request.urlopen(url) as response:
        total = int(response.info().get("Content-Length", -1))
        downloaded = 0
        block_size = 8192  # 8KB

        with open(zip_path, "wb") as out_file:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break

                out_file.write(buffer)
                downloaded += len(buffer)

                if total > 0:
                    percent = downloaded / total * 100
                    print(f"\r[init_data] 다운로드 진행률: {percent:5.1f}% ({downloaded}/{total} bytes)", end="")

    print(f"\n[init_data] 다운로드 완료: {zip_path}")
    return True

def unzip_if_needed(zip_path: str, extract_to: str):
    """폴더가 비었으면 ZIP을 압축 해제"""
    if os.path.exists(zip_path):
        print(f"[init_data] {extract_to}에 ZIP 파일 존재 → ZIP 압축 해제 시작")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            total_files = len(file_list)

            for i, file in enumerate(file_list, start=1):
                zip_ref.extract(file, extract_to)

                percent = i / total_files * 100
                print(f"\r[init_data] 압축 해제 진행률: {percent:5.1f}% ({i}/{total_files})", end="")

        print(f"\n[init_data] 압축 해제 완료: {extract_to}")

        try:
            os.remove(zip_path)
            print(f"[init_data] ZIP 삭제 완료: {zip_path}")
        except Exception as e:
            print(f"[init_data] ZIP 삭제 실패: {e}")

    else:
        print(f"[init_data] {extract_to}에 ZIP 파일 없음 → unzip 생략")
    
    macos_junk = os.path.join(extract_to, "__MACOSX")
    if os.path.exists(macos_junk):
        shutil.rmtree(macos_junk, ignore_errors=True)
        print("[init_data] __MACOSX 제거 완료")


def init_data():
    """데이터 ZIP과 weight ZIP을 자동 다운로드 + 압축해제"""
    print("\n==== [INIT_DATA] 데이터 및 모델 파일 검사 시작 ====\n")

    # 1) 폴더 준비
    ensure_dir(DATA_DIR)
    ensure_dir(WEIGHTS_DIR)

    data_ready_dir = os.path.join(DATA_DIR, "animelist-dataset")
    weights_ready_dir = os.path.join(WEIGHTS_DIR, "weights")

    # 2) 데이터 디렉터리 검사 후 없으면 ZIP 다운로드 + 압축 해제
    if not os.path.isdir(data_ready_dir) or not os.listdir(data_ready_dir):
        print(f"[init_data] {data_ready_dir} 비어있음 또는 없음 → 데이터 번들 처리 시작")
        if DATA_BUNDLE_URL:
            download_if_missing(DATA_BUNDLE_URL, DATA_ZIP_PATH)
            unzip_if_needed(DATA_ZIP_PATH, DATA_DIR)
        else:
            print("[init_data] DATA_BUNDLE_URL 없음 → 데이터 다운로드 스킵")
    else:
        print(f"[init_data] {data_ready_dir} 이미 존재 → 데이터 다운로드/압축 해제 생략")

    # 3) 모델 weight 디렉터리 검사 후 없으면 ZIP 다운로드 + 압축 해제
    if not os.path.isdir(weights_ready_dir) or not os.listdir(weights_ready_dir):
        print(f"[init_data] {weights_ready_dir} 비어있음 또는 없음 → 모델 번들 처리 시작")
        if WEIGHTS_BUNDLE_URL:
            download_if_missing(WEIGHTS_BUNDLE_URL, WEIGHTS_ZIP_PATH)
            unzip_if_needed(WEIGHTS_ZIP_PATH, WEIGHTS_DIR)
        else:
            print("[init_data] WEIGHTS_BUNDLE_URL 없음 → 모델 다운로드 스킵")
    else:
        print(f"[init_data] {weights_ready_dir} 이미 존재 → 모델 다운로드/압축 해제 생략")

    print("\n==== [INIT_DATA] 데이터 준비 완료 ====\n")