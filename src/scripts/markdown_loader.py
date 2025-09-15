import os
import pathlib

def load_markdown_file(file_path):
    """
    Load the content of a markdown file using os and pathlib.
    
    Args:
        file_path (str or Path): Path to the markdown file
        
    Returns:
        str: Content of the markdown file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If there's an error reading the file
    """
    path = pathlib.Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Markdown file not found: {file_path}")
    try:
        # Use os to get absolute path (demonstrating os usage)
        abs_path = os.path.abspath(str(path))
        with open(abs_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except Exception as e:
        raise IOError(f"Error reading file {file_path}: {e}")