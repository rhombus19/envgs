import os
import subprocess
import sys
from pathlib import Path

def extract_frames(folder):
    folder_path = Path(folder)
    if not folder_path.is_dir():
        print(f"Error: {folder} is not a valid directory.")
        return

    for subfolder in folder_path.iterdir():
        if not subfolder.is_dir(): continue
        for file in subfolder.iterdir():
            if file.suffix.lower() in [".mp4", ".MOV", ".mov"]:
                output_pattern = "frame%05d.png"
                out_folder = Path(subfolder) / (file.stem + "_frames")
                out_folder.mkdir(exist_ok=True, parents=True)
                cmd = [
                    "ffmpeg",
                    "-i", str(file.resolve()),
                    "-vf",
                    "select='not(mod(n\\,10))'," # Select every 10th frame
                    "zscale=matrixin=bt2020nc:transferin=arib-std-b67:primariesin=bt2020:rangein=tv:t=linear:npl=100,"  #chatgpt estimated values for IPhone footage tonemapping
                    "tonemap=tonemap=hable:param=0.5:desat=0,"
                    "zscale=matrix=bt709:transfer=bt709:primaries=bt709:range=tv:t=bt709,"
                    "format=rgb24",
                    "-vsync", "vfr",
                    "-pix_fmt", "rgb24",
                    str(out_folder.resolve() / output_pattern),
                ]

                print(f"Running: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_frames.py <folder>")
        sys.exit(1)
    extract_frames(sys.argv[1])