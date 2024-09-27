# %%
import os

# %%

def list_python_files(directory):
    try:
        files = [f for f in os.listdir(directory) if f.endswith('.py')]
        return files
    except FileNotFoundError:
        return f"The directory {directory} does not exist."

directory = '/home/luismiguel/bitbucket/diffractio/diffractio/'
python_files = list_python_files(directory)



# print("Python files in the directory:")
# for file in python_files:
#     print(file)
    
    
  


# %%

 
 
def list_classes_and_functions(file_path):
    """
    list_classes_and_functions _summary_

    _extended_summary_

    Args:
        file_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    classes = []
    functions = []
    
    with open(file_path, 'r') as file:
        file_content = file.read()

    # Extract classes and functions without using exec
    for line in file_content.splitlines():
        if line.strip().startswith('class '):
            class_name = line.split()[1].split('(')[0]
            classes.append(class_name)
        elif line.strip().startswith('def '):
            function_name = line.split()[1].split('(')[0]
            functions.append(function_name)
    
    # Initialize class_functions after extracting class names
    class_functions = {class_name: [] for class_name in classes}
    standalone_functions = []

    current_class = None
    for line in file_content.splitlines():
        if line.strip().startswith('class '):
            current_class = line.split()[1].split('(')[0]
        elif line.strip().startswith('def '):
            function_name = line.split()[1].split('(')[0]
            if current_class:
                class_functions[current_class].append(function_name)
            else:
                standalone_functions.append(function_name)

    return classes, standalone_functions, class_functions
    
    


# %%
directory = '/home/luismiguel/bitbucket/diffractio/diffractio/'
file = 'scalar_masks_XY.py'

file_path = os.path.join(directory, file)
classes, standalone_functions, class_functions = list_classes_and_functions(file_path)



# %%

# for class_name in classes:
#     print(f"Class: {class_name}")
# for function_name in standalone_functions:
#     print(f"Standalone Function: {function_name}")
# for class_name, functions in class_functions.items():
#     print(f"Class {class_name} Functions: {functions}")
    
    


# %%

def print_data_from_file(file_path) -> None:
    classes, standalone_functions, class_functions = list_classes_and_functions(file_path)
    print("Classes:")
    for cl in classes:
        print(cl)
        for function in class_functions[cl]:
            print("     {}".format(function))
    print("Standalone functions:")
    for function in standalone_functions:
        print(function)



# %%
# print_data_from_file(file_path)

# %%


# print("Python files in the directory:")
# for file in python_files:
#     print(file)
#     file_path = os.path.join(directory, file)
#     classes, standalone_functions, class_functions = list_classes_and_functions(file_path)
#     print_data_from_file(file_path)
#     print("\n\n")
    


# # %%

#  with open('output.txt', 'w') as rst_file:
#         rst_file.write("Python files in the directory:\n")
#         for file in python_files:
#             rst_file.write(f"{file}\n")
#             file_path = os.path.join(directory, file)
#             classes, standalone_functions, class_functions = list_classes_and_functions(file_path)
#             rst_file.write("Classes:\n")
#             for cl in classes:
#                 rst_file.write(f"{cl}\n")
#                 for function in class_functions[cl]:
#                     rst_file.write(f"    {function}\n")
#             rst_file.write("Standalone functions:\n")
#             for function in standalone_functions:
#                 rst_file.write(f"{function}\n")
#             rst_file.write("\n\n")


# # %%

# with open('output.txt', 'w') as rst_file:
#     rst_file.write("Python files in the directory:\n")
#     for file in python_files:
#         rst_file.write(f"{file}\n")
#         file_path = os.path.join(directory, file)
#         classes, standalone_functions, class_functions = list_classes_and_functions(file_path)
#         rst_file.write("Classes:\n")
#         for cl in classes:
#             rst_file.write(f"{cl}\n")
#             for function in class_functions[cl]:
#                 rst_file.write(f"    {function}\n")
#         rst_file.write("Standalone functions:\n")
#         for function in standalone_functions:
#             rst_file.write(f"{function}\n")
#         rst_file.write("\n\n")
#     rst_file.write(f"Total number of Python files: {len(python_files)}\n")
#     for file in python_files:
#         file_path = os.path.join(directory, file)
#         classes, standalone_functions, class_functions = list_classes_and_functions(file_path)
#         rst_file.write(f"\nFile: {file}\n")
#         rst_file.write(f"Number of classes: {len(classes)}\n")
#         for cl in classes:
#             rst_file.write(f"Class: {cl}, Number of functions: {len(class_functions[cl])}\n")

# %%

num_functions = 0

with open('functions.rst', 'w') as rst_file:
    rst_file.write("Python files in the directory:\n")
    rst_file.write("================================\n\n")
    for file in sorted(python_files):
        rst_file.write(f"{file}\n")
        rst_file.write("__________________________________________________\n\n")

        file_path = os.path.join(directory, file)
        classes, standalone_functions, class_functions = list_classes_and_functions(file_path)
        for cl in classes:
            if cl not in ('', [], None):
                rst_file.write(f" Class: **{cl}**\n")
                rst_file.write(f"   Number of functions: {len(class_functions[cl])}\n\n")
                num_functions += len(class_functions[cl])

            for function in sorted(class_functions[cl]):
                rst_file.write(f"    - {function}\n\n")
                
        rst_file.write("\n Standalone functions:\n\n")
        rst_file.write(f"   Number of functions: {len(standalone_functions)}\n\n")
        num_functions += len(standalone_functions)

        for function in sorted(standalone_functions):
            rst_file.write(f"  - {function}\n\n")
        rst_file.write("\n\n\n\n")
        
        
    rst_file.write(f"Python files and lines\n")
    rst_file.write(f"============================\n\n")
  
    
    ## Number of lines in each file

    for file in python_files:
        file_path = os.path.join(directory, file)
        classes, standalone_functions, class_functions = list_classes_and_functions(file_path)
        rst_file.write(f"\nFile: {file}\n")
        rst_file.write(f"_____________________________________________________________________\n\n")

        with open(file_path, 'r') as f:
            lines = f.readlines()
            num_lines = len(lines)
            rst_file.write(f"  Number of lines: {num_lines}\n\n")
        rst_file.write(f"  Number of classes: {len(classes)}\n\n")
        for cl in classes:
            rst_file.write(f"    Class: {cl}, Number of functions: {len(class_functions[cl])}\n\n")
            

    total_lines = sum(len(open(os.path.join(directory, file), 'r').readlines()) for file in python_files)
    
    
    rst_file.write(f"Total\n")
    rst_file.write(f"============================\n\n")
    rst_file.write(f" Total number of Python files: {len(python_files)}\n\n")
    rst_file.write(f" Total number of functions: {num_functions}\n\n")
    rst_file.write(f" Total number of lines across all files: {total_lines}\n\n")
    
