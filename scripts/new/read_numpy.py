import numpy as np
import argparse
import os

def load_npy(path):
    data = np.load(path, allow_pickle=True)
    print(f"File: {path}")
    print(f"Type: {type(data)}")
    print(f"Shape: {getattr(data, 'shape', None)}")
    print(f"Dtype: {getattr(data, 'dtype', None)}")
    print("\nPreview:")
    print(data)
    return data

def load_npz(path):
    archive = np.load(path, allow_pickle=True)
    print(f"File: {path}")
    print("Keys:", archive.files)
    print()

    for k in archive.files:
        arr = archive[k]
        print(f"--- Key: {k} ---")
        print(f"  Shape: {arr.shape}")
        print(f"  Dtype: {arr.dtype}")
        print("  Preview:")
        print(arr)
        print()

    return archive

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to .npy or .npz file")
    args = parser.parse_args()

    path = args.path
    if not os.path.exists(path):
        print("File not found:", path)
        return

    if path.endswith(".npy"):
        load_npy(path)
    elif path.endswith(".npz"):
        load_npz(path)
    else:
        print("Unsupported file type:", path)

if __name__ == "__main__":
    main()
