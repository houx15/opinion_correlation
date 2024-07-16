from configs import *
from utils.utils import extract_7z_files


def get_bangdan_files_dir(year):
    return f"{DATA_SOURCE_DIR}/{year}/bangdan/"


def get_bangdan_unzipped_files_dir(year):
    return f"bangdan_data/{year}/"


def unzip_all_bangdan_files():
    for year in ANALYSIS_YEARS:
        bangdan_files_dir = get_bangdan_files_dir(year)
        unzipped_dir = get_bangdan_unzipped_files_dir(year)
        extract_7z_files(source_folder=bangdan_files_dir, target_folder=unzipped_dir)


if __name__ == "__main__":
    unzip_all_bangdan_files()
