
import os

def list_python_files_and_line_counts(directory: str) -> list[tuple[str, int]]:
    """List Python files and their line counts in a given directory.

    Args:
        directory (str): The directory to search for Python files.

    Returns:
        list[tuple[str, int]]: A list of tuples containing the file name and line count.
    """

    num_lines = 0
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)
                python_files.append((file, line_count))
                num_lines = num_lines + line_count
    return python_files, num_lines

# Example usage
directory_path = "/media/luismiguel/mas_datos/bitbucket/diffractio/diffractio"
python_files, num_lines = list_python_files_and_line_counts(directory_path)
for file_name, line_count in python_files:
    print(f"{file_name}: {line_count} lines")

print("Total number of lines in Python files:", num_lines)