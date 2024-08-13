import os

def load_prompt(file: str):
    """
    Loads the content of a .txt file into a string.
    ------------------------------------------------
    Args:
     file (str): The path to the .txt file to be loaded.
    
    Raises:
     AssertionError: If the file extension is not .txt.
     Exception: If the file cannot be read, with an additional message indicating the specific file.
    
    Returns:
     str: The content of the file as a string.
    """

    _, ext = os.path.splitext(file)

    assert ext == ".txt", "Extension not allowed. Valid extention .txt" 

    template  = ""
    try:
        with open(file, 'r') as f:
            template += str(f.read())+"\n"
    except Exception as e:
        print(e)
        raise Exception (f"Could not read file {file}")
    return template