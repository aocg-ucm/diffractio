
import os

def list_python_files_and_line_counts(directory: str, ext: str) -> list[tuple[str, int]]:
    """List Python files and their line counts in a given directory.

    Args:
        directory (str): The directory to search for Python files.

    Returns:
        list[tuple[str, int]]: A list of tuples containing the file name and line count.
    """

    num_lines = 0
    python_files = []
    
    for root, _, files in os.walk(directory):
        for file in sorted(files):
            if file.endswith(ext) and not file.startswith("_"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)
                python_files.append((file, line_count))
                num_lines = num_lines + line_count
    return python_files, num_lines


directory_path = "../diffractio"
python_files, num_lines = list_python_files_and_line_counts(directory_path, ".py")

with open("docs/files_python.rst", 'w') as f:
    
    text = f"Python files"
    print(text)
    f.write(text+"\n")
    f.write("================================\n\n")

    for file_name, line_count in sorted(python_files):
        text = f" - {file_name}: {line_count} lines"
        print(text)
        f.write(text+"\n")


    text = f"\nTotal number of lines in Python files: {num_lines}"
    print(text)
    f.write(text+"\n")



directory_path = "../diffractio/docs/source"
python_files, num_lines = list_python_files_and_line_counts(directory_path, ".ipynb")



with open("docs/files_jupyter.rst", 'w') as f:
    text = f"Jupyter files"
    print(text)
    f.write(text+"\n")
    f.write("================================\n\n")

    for file_name, line_count in sorted(python_files):
        text = f" - {file_name}: {line_count} lines"
        print(text)
        f.write(text+"\n")


    text = f"\nTotal number of lines in Python files: {num_lines}"
    print(text)
    f.write(text+"\n")
