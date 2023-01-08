import os
from pathlib import Path

from tqdm import tqdm
from natsort import natsorted

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# TODO cover by tests
def iterate_with_structure(in_folder, out_folder, supported_extensions=IMG_EXTENSIONS):
    in_folder = Path(in_folder)
    out_folder = Path(out_folder)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)
    files = []
    for pattern in supported_extensions:
        files.extend(in_folder.rglob(pattern='*' + pattern))

    for file_path in tqdm(natsorted(files)):
        sub_path = os.path.relpath(file_path, in_folder)
        new_path = out_folder / sub_path
        if not os.path.exists(new_path.parent):
            new_path.parent.mkdir(exist_ok=True, parents=True) # Create parent folder

        yield (file_path, new_path)


def iterate_recursively(in_folder, supported_extensions=IMG_EXTENSIONS):
    in_folder = Path(in_folder)
    files = []
    for pattern in supported_extensions:
        files.extend(in_folder.rglob(pattern='*' + pattern))

    for file_path in files:
        yield file_path

if __name__ == '__main__':
    for (in_path, out_path) in iterate_with_structure('/home/malchul/Documents', 'out_fld'):
        print(in_path, out_path)
