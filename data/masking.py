from nilearn.image import load_img, resample_to_img, new_img_like, get_data
from nibabel.nifti1 import Nifti1Image
from pathlib import Path
import numpy as np


def apply_mask(filtered_imgs, mask_path: Path) -> Nifti1Image:
    """apply mask to images by resampling the mask to the image space.

    parameters:
    filtered_imgs : nibabel.Nifti1Image
        4d image containing beta images or volumes to be masked.
    mask_path : Path
        path to a nifti mask file to be resampled to `filtered_imgs` space.

    returns:
    Nifti1Image
        resampled mask image aligned to `filtered_imgs`.
    """

    mask_img = load_img(mask_path)

    # reshape mask to fit train data
    masked_imgs = resample_to_img(mask_img, filtered_imgs, interpolation="nearest")
    print(f"{masked_imgs=}")
    print(f"{masked_imgs.shape=}")
    print(f"{filtered_imgs.shape=}")

    return masked_imgs


def get_process_mask(
    process_mask_path: Path | None, masked_imgs: Nifti1Image
) -> Nifti1Image | None:
    """load and resample a process mask to match masked images.

    parameters:
    process_mask_path : Path | None
        optional path to a process mask nifti file. if none, None is returned.
    masked_imgs : nibabel.Nifti1Image
        reference image used to resample the process mask.

    returns:
    Nifti1Image | None
        resampled process mask or None if no path supplied.
    """

    if process_mask_path:
        process_mask = load_img(process_mask_path)
        process_mask = resample_to_img(
            process_mask, masked_imgs, interpolation="nearest"
        )
        print(f"{process_mask.shape=}")
        print(f"{process_mask=}")
        return process_mask
    else:
        return None


def create_single_slice_mask(
    mask_path: Path,
    slice_index: int,
    axis: int = 2,
    save_path: Path | None = None,
) -> Nifti1Image:
    """create a binary mask that selects a single slice from the input mask.

    parameters:
    mask_path : Path
        path to the input nifti mask to derive the single-slice mask from.
    slice_index : int
        index of the slice to keep (along `axis`).
    axis : int
        axis along which to select the slice (0, 1, or 2).
    save_path : Path | None
        optional path to save the generated mask; if provided the mask is written.

    returns:
    Nifti1Image
        binary mask nifti image with only the requested slice set to 1.
    """

    mask_img = load_img(mask_path)
    print(
        f"mask shape: {mask_img.shape}, non-zero voxels: {np.sum(get_data(mask_img).astype(int))}"
    )
    # .astype() makes a copy.
    process_mask = get_data(mask_img).astype(int)
    process_mask[..., (slice_index + 1) :] = 0
    process_mask[..., :slice_index] = 0
    process_mask_img = new_img_like(mask_img, process_mask)

    # optionally save the mask
    if save_path:
        process_mask_img.to_filename(save_path)
        print(f"process mask saved to {save_path}")
    print(f"created single-slice mask: slice {slice_index} on axis {axis}")
    print(
        f"mask shape: {process_mask_img.shape}, non-zero voxels: {np.sum(get_data(process_mask_img).astype(int))}"
    )

    return process_mask_img
