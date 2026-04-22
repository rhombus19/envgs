from easyvolcap.utils.colmap_utils import read_images_binary, write_images_binary
from pathlib import Path

src = Path("/home/roman/ba/niederfuellbach2_autos/niederfuellbach2_datasets")
for ds in src.iterdir():

    sparse_img_p = ds / "colmap_sparse" /  "0" / "images.bin"
    images = read_images_binary(sparse_img_p)
    image_files = Path(ds / "images")
    
    image_names = sorted([p.name for p in image_files.iterdir()])
    
    for k in sorted(images.keys()):
        images[k].name = image_names[k-1]

    write_images_binary(images, sparse_img_p)