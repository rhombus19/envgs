import argparse
import subprocess
from pathlib import Path

def collect_videos(path, video_exts):
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    if path.is_file():
        if path.suffix.lower() not in video_exts:
            raise ValueError(
                f"{path} is not a supported video file (supported: {sorted(video_exts)})"
            )
        return [path]

    if path.is_dir():
        # Recursive scan keeps compatibility with directory trees.
        return sorted(
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in video_exts
        )

    raise ValueError(f"{path} is neither a file nor a directory")


from pathlib import Path
import subprocess


def extract_frames(path, n, hdr=False):
    if n <= 0:
        raise ValueError("--n must be a positive integer")

    source_path = Path(path)

    video_exts = {".mp4", ".mov"}
    video_files = collect_videos(source_path, video_exts)
    if not video_files:
        print(f"No supported videos found at {source_path}.")
        return

    if hdr:
        vf_chain = (
            f"select='not(mod(n\\,{n}))',"
            "zscale=matrixin=bt2020nc:transferin=arib-std-b67:primariesin=bt2020:rangein=tv:t=linear:npl=100,"
            "tonemap=tonemap=hable:param=0.5:desat=0,"
            "zscale=matrix=bt709:transfer=bt709:primaries=bt709:range=tv:t=bt709,"
            "scale='if(gte(iw,ih),1920,1080)':'if(gte(iw,ih),1080,1920)':flags=lanczos,"
            "setsar=1"
        )
    else:
        vf_chain = (
            f"select='not(mod(n\\,{n}))',"
            "scale='if(gte(iw,ih),1920,1080)':'if(gte(iw,ih),1080,1920)':flags=lanczos,"
            "setsar=1"
        )

    for file in video_files:
        output_pattern = "frame%05d.png"
        suffix = "hdr2sdr" if hdr else "sdr"
        out_folder = file.parent / "frames" / f"{file.stem}_frames_{n}_{suffix}"
        out_folder.mkdir(exist_ok=True, parents=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(file.resolve()),
            "-vf", vf_chain,
            "-fps_mode", "vfr",
            str(out_folder.resolve() / output_pattern),
        ]

        print(f"Running: {' '.join(cmd)}")
        # subprocess.run(cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Frame Sampler')
    parser.add_argument(
        '--videos_path',
        required=True,
        help='Path to a video file or a directory containing videos to be sampled',
    )
    parser.add_argument('--n', type=int, help='Sample every n frames', default=1)
    parser.add_argument('--hdr', action="store_true", help='Input video is hdr, frame need to be converted to sdr', default=False)
    args = parser.parse_args()

    extract_frames(args.videos_path, args.n, args.hdr)
