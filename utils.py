"""
Utility functions for RepoMap.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List

try:
    import tiktoken
except ImportError:
    print("Error: tiktoken is required. Install with: pip install tiktoken")
    sys.exit(1)

# Finding #8: cache tiktoken encoding objects
_ENCODING_CACHE = {}

def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """Count tokens in text using tiktoken."""
    if not text:
        return 0

    if model_name not in _ENCODING_CACHE:
        try:
            _ENCODING_CACHE[model_name] = tiktoken.encoding_for_model(model_name)
        except KeyError:
            _ENCODING_CACHE[model_name] = tiktoken.get_encoding("cl100k_base")

    return len(_ENCODING_CACHE[model_name].encode(text))


IGNORED_DIRS = {'node_modules', '__pycache__', 'venv', 'env', '.venv', 'dist', 'build'}


def find_src_files(directory: str) -> List[str]:
    """Find source files in a directory using os.scandir (fast, symlink-safe)."""
    if not os.path.isdir(directory):
        return [directory] if os.path.isfile(directory) else []
    src_files = []
    root_realpath = os.path.realpath(directory)
    queue = [directory]
    while queue:
        curr_dir = queue.pop()
        try:
            for entry in os.scandir(curr_dir):
                if entry.name.startswith('.'):
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if entry.name not in IGNORED_DIRS:
                        queue.append(entry.path)
                elif entry.is_file(follow_symlinks=False):
                    src_files.append(entry.path)
                elif entry.is_symlink():
                    try:
                        if os.path.realpath(entry.path).startswith(root_realpath):
                            if entry.is_file():
                                src_files.append(entry.path)
                    except OSError:
                        continue
        except OSError:
            continue
    return src_files


def read_text(filename: str, encoding: str = "utf-8", silent: bool = False) -> Optional[str]:
    """Read text from file with error handling."""
    try:
        return Path(filename).read_text(encoding=encoding, errors='ignore')
    except FileNotFoundError:
        if not silent:
            print(f"Error: {filename} not found.")
        return None
    except IsADirectoryError:
        if not silent:
            print(f"Error: {filename} is a directory.")
        return None
    except OSError as e:
        if not silent:
            print(f"Error reading {filename}: {e}")
        return None
    except UnicodeError as e:
        if not silent:
            print(f"Error decoding {filename}: {e}")
        return None
    except Exception as e:
        if not silent:
            print(f"An unexpected error occurred while reading {filename}: {e}")
        return None
