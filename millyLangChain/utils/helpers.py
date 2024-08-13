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


def printr(text:str):
    """Prints the given text in red color."""
    print("\033[91m{}\033[0m".format(text))

def printy(text:str):
    """Prints the given text in yellow color."""
    print("\033[93m{}\033[0m".format(text))

def printg(text:str):
    """Prints the given text in green color."""
    print("\033[92m{}\033[0m".format(text))

def printb(text: str):
    """Prints the given text in blue color."""
    print("\033[94m{}\033[0m".format(text))

def printc(text: str):
    """Prints the given text in cyan color."""
    print("\033[36m{}\033[0m".format(text))