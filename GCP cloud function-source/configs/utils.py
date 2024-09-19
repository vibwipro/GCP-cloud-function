
def file_type_from_name(file_path: str) -> str:
    # expected file_path:
    # file_type/country_code/filename.extension
    return file_path.split('/')[-1].lower()