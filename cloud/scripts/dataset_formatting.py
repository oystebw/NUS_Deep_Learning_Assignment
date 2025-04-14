from typing import List, Tuple
import os
from pathlib import Path
import argparse
import random

def build_argparser() -> argparse.ArgumentParser:
    """
    Build the argument parser for command line arguments.
    Returns:
        argparse.ArgumentParser: The argument parser.
    """
    parser = argparse.ArgumentParser(description="Process real clouds dataset.")
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing the images to process."
    )
    parser.add_argument(
        "--A", type=str, required=True,
        help="Directory containing the images of modality A to process."
    )
    parser.add_argument(
        "--B", type=str, required=True,
        help="Directory containing the images of modality B to process."
    )
    parser.add_argument(
        "--folders", type=str, nargs=4, required=True,
        help="List of folder names to create for training and testing data."
    )
    parser.add_argument(
        "--alpha", type=float, required=True,
        help="Ratio of training data to total data."
    )
    return parser


def browse_folder(
        path: Path,
        A: str,
        B: str) -> Tuple[List[Path], List[Path]]:
    """
    Browse a directory and return a list of all the png files in it and its subfolder.
    """
    if isinstance(path, str):
        path = Path(path)
    A_dir = path / A
    B_dir = path / B
    A_paths = sorted(list(A_dir.glob("*.png"))) + list(A_dir.glob("*.jpg"))
    B_paths = sorted(list(B_dir.glob("*.png"))) + list(B_dir.glob("*.jpg"))
    return A_paths, B_paths


def check_existence(
        filenames: List[Path]) -> None:
    """
    Check if the paths exist.
    Args:
        filenames (List[Path]): List of filenames.
    Returns:
        None
    """
    for f in filenames:
        if not f.exists():
            raise FileNotFoundError(f"[!] File does not exist: {f}")


def create_folders(
        input_dir: Path,
        folder_list: List[str]) -> None:
    """
    Create folders for training and testing data.
    Args:
        input_dir (Path): Input directory.
        folder_list (List[str]): List of folder names to create.
    Returns:
        None
    """
    for folder in folder_list:
        full_path = input_dir / folder
        full_path.mkdir(parents=True, exist_ok=True)


def split_train_test(
        filenames: List[Path],
        alpha: float) -> Tuple[List[Path], List[Path]]:
    """
    Split the filenames into training and testing sets.
    Args:
        filenames (List[Path]): List of filenames.
        alpha (float): Ratio of training data to total data.
    Returns:
        Tuple[List[Path], List[Path]]: Training and testing filenames.
    """
    random.shuffle(filenames)
    split_idx = int(len(filenames) * alpha)
    return filenames[:split_idx], filenames[split_idx:]

def create_symlinks(
        filenames: List[Path],
        input_dir: Path,
        split_dir: Path) -> None:
    """
    Create symbolic links for the training and testing images.
    Args:
        filenames (List[Path]): List of filenames.
        input_dir (Path): Input directory.
        split_dir (Path): Split directory.
    Returns:
        None
    """
    for f in filenames:
        link_name = split_dir / f.name
        if link_name.exists():
            link_name.unlink()
        os.symlink(f.resolve(), link_name)

def process(input_dir: str, A: str, B: str, folders: List[str], alpha: float) -> None:
    """
    Format the dataset for training and testing.
    Args:
        data_dir (str): Directory containing the images.
        A (str): Name of the first dataset.
        B (str): Name of the second dataset.
        folders (List[str]): List of folder names to create.
        alpha (float): Ratio of training data to total data.
    Returns:
        None
    """
    base = Path(input_dir)
    A_files, B_files = browse_folder(base, A, B)
    check_existence(A_files + B_files)
    create_folders(base, folders)

    A_train, A_test = split_train_test(A_files, alpha)
    B_train, B_test = split_train_test(B_files, alpha)

    create_symlinks(A_train, base / A, base / folders[0])  # trainA
    create_symlinks(B_train, base / B, base / folders[1])  # trainB
    create_symlinks(A_test, base / A, base / folders[2])   # testA
    create_symlinks(B_test, base / B, base / folders[3])   # testB


def main():
    """
    Main function to execute the script.
    """
    parser = build_argparser()
    args = parser.parse_args()

    process(args.input_dir, args.A, args.B, args.folders, args.alpha)

if __name__ == "__main__":
    main()
