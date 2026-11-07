import numpy as np
from skimage.morphology import remove_small_objects
import torch
import nibabel as nib
from skimage import transform
from skimage.morphology import (
    label,
    remove_small_objects,
    remove_small_holes,
    closing,
    ball,
)
import scipy.ndimage as ndi
import SimpleITK as sitk
from dataclasses import dataclass

from skimage import filters, measure


def remap_mask(mask, mapping):
    out = np.zeros_like(mask)
    for src, dst in mapping.items():
        out[mask == src] = dst
    return out


def anisodiff3(
    stack,
    niter=1,
    kappa=50,
    gamma=0.1,
    step=(1.0, 1.0, 1.0),
    option=1,
):
    """
    3D Anisotropic diffusion.

    Usage:
    stackout = anisodiff(stack, niter, kappa, gamma, option)

    Arguments:
            stack  - input stack
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (z,y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the middle z-plane will be plotted on every
                 iteration

    Returns:
            stackout   - diffused stack.
    """

    # initialize output array
    stack = stack.astype("float32")
    stackout = stack.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(stackout)
    deltaE = deltaS.copy()
    deltaD = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    UD = deltaS.copy()
    gS = np.ones_like(stackout)
    gE = gS.copy()
    gD = gS.copy()

    for ii in range(niter):

        # calculate the diffs
        deltaD[:-1, :, :] = np.diff(stackout, axis=0)
        deltaS[:, :-1, :] = np.diff(stackout, axis=1)
        deltaE[:, :, :-1] = np.diff(stackout, axis=2)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gD = np.exp(-((deltaD / kappa) ** 2.0)) / step[0]
            gS = np.exp(-((deltaS / kappa) ** 2.0)) / step[1]
            gE = np.exp(-((deltaE / kappa) ** 2.0)) / step[2]
        elif option == 2:
            gD = 1.0 / (1.0 + (deltaD / kappa) ** 2.0) / step[0]
            gS = 1.0 / (1.0 + (deltaS / kappa) ** 2.0) / step[1]
            gE = 1.0 / (1.0 + (deltaE / kappa) ** 2.0) / step[2]

        # update matrices
        D = gD * deltaD
        E = gE * deltaE
        S = gS * deltaS

        # subtract a copy that has been shifted 'Up/North/West' by one
        # pixel. don't as questions. just do it. trust me.
        UD[:] = D
        NS[:] = S
        EW[:] = E
        UD[1:, :, :] -= D[:-1, :, :]
        NS[:, 1:, :] -= S[:, :-1, :]
        EW[:, :, 1:] -= E[:, :, :-1]

        # update the image
        stackout += gamma * (UD + NS + EW)

    return stackout


def anisodiff(
    img,
    niter=1,
    kappa=50,
    gamma=0.1,
    step=(1.0, 1.0),
    option=1,
):

    # initialize output array
    img = img.astype("float32")
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    for ii in range(niter):

        # calculate the diffs
        deltaS[:-1, :] = np.diff(imgout, axis=0)
        deltaE[:, :-1] = np.diff(imgout, axis=1)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-((deltaS / kappa) ** 2.0)) / step[0]
            gE = np.exp(-((deltaE / kappa) ** 2.0)) / step[1]
        elif option == 2:
            gS = 1.0 / (1.0 + (deltaS / kappa) ** 2.0) / step[0]
            gE = 1.0 / (1.0 + (deltaE / kappa) ** 2.0) / step[1]

        # update matrices
        E = gE * deltaE
        S = gS * deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel.
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]

        # update the image
        imgout += gamma * (NS + EW)

    return imgout


def n4_bias_correction(image: np.ndarray, shrink_factor: int = 2) -> np.ndarray:
    """
    N4 bias correction for 3D or 4D CMR images. For 4D images,
    the bias field is estimated from the mean frame and applied to all frames.

    Args:
        image: 3D array (W, H, Z) or 4D array (W, H, Z, T)
        shrink_factor: shrink factor for N4 bias correction (default: 2)

    Returns:
        Bias-corrected array
    """
    if image.ndim == 3:
        """N4 cho single 3D volume (W, H, Z)"""
        itk = sitk.GetImageFromArray(image.astype(np.float32))
        itk_shrunk = sitk.Shrink(itk, [1, shrink_factor, shrink_factor])
        mask_shrunk = sitk.OtsuThreshold(itk_shrunk, 0, 1, 200)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
        corrector.Execute(itk_shrunk, mask_shrunk)

        log_bias = corrector.GetLogBiasFieldAsImage(itk_shrunk)
        log_bias_full = sitk.Resample(log_bias, itk)

        corrected = sitk.Divide(itk, sitk.Exp(log_bias_full))
        return sitk.GetArrayFromImage(corrected).astype(np.float32)

    elif image.ndim == 4:

        mean_frame = image.mean(axis=-1)  # (W, H, Z)

        itk = sitk.GetImageFromArray(mean_frame.astype(np.float32))
        itk_shrunk = sitk.Shrink(itk, [1, shrink_factor, shrink_factor])
        mask_shrunk = sitk.OtsuThreshold(itk_shrunk, 0, 1, 200)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
        corrector.Execute(itk_shrunk, mask_shrunk)

        log_bias = corrector.GetLogBiasFieldAsImage(itk_shrunk)
        log_bias_full = sitk.Resample(log_bias, itk)
        bias_arr = np.exp(sitk.GetArrayFromImage(log_bias_full).astype(np.float32))  # (W, H, Z)

        result = image.astype(np.float32) / (bias_arr[..., np.newaxis] + 1e-8)
        return result

    else:
        raise ValueError(f"Expected 3D or 4D array, got shape {image.shape}")


def preprocess_cmr(
    volume: np.ndarray,
) -> np.ndarray:
    enhanced = anisodiff3(volume, niter=1, kappa=50, gamma=0.015, step=(1.0, 1.0, 1.0), option=1)
    enhanced = n4_bias_correction(enhanced)
    enhanced = np.clip(enhanced, np.percentile(enhanced, 0.5), np.percentile(enhanced, 99.5))
    enhanced = (enhanced - np.min(enhanced)) / (np.max(enhanced) - np.min(enhanced))
    return enhanced


def augment_scar_only_elastic(
    image,
    mask,
    scar_label=3,
    myo_label=2,
    alpha=25.0,
    sigma=8.0,
    roi_dilation=15,
    myo_dilation=2,
    boundary_sigma=1.5,
):
    """
    Elastic augmentation for myocardial scar only.

    image: 2D image, shape [H, W]
    mask:  2D mask, shape [H, W]
           1 = LV cavity
           2 = LV myocardium
           3 = scar
           4 = RV cavity

    Returns:
        aug_image, aug_mask
    """

    image = image.astype(np.float32)
    mask = mask.copy()

    scar = mask == scar_label
    myo = mask == myo_label

    if scar.sum() == 0:
        return image.copy(), mask.copy()

    h, w = image.shape

    # --------------------------------------------------
    # 1. Create random smooth deformation field
    # --------------------------------------------------
    dx = np.random.randn(h, w).astype(np.float32)
    dy = np.random.randn(h, w).astype(np.float32)

    dx = ndi.gaussian_filter(dx, sigma=sigma)
    dy = ndi.gaussian_filter(dy, sigma=sigma)

    dx *= alpha
    dy *= alpha

    # --------------------------------------------------
    # 2. Restrict deformation to scar neighborhood
    # --------------------------------------------------
    roi = ndi.binary_dilation(scar, iterations=roi_dilation)

    dx *= roi
    dy *= roi

    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    coords = np.array([yy + dy, xx + dx])

    # --------------------------------------------------
    # 3. Warp scar mask
    # --------------------------------------------------
    scar_def = (
        ndi.map_coordinates(
            scar.astype(np.float32),
            coords,
            order=1,
            mode="constant",
            cval=0.0,
        )
        > 0.5
    )

    # --------------------------------------------------
    # 4. Keep deformed scar anatomically valid
    #    Scar should remain inside/near myocardium.
    # --------------------------------------------------
    myo_region = ndi.binary_dilation(myo | scar, iterations=myo_dilation).astype(bool)
    scar_def = scar_def & myo_region

    if scar_def.sum() == 0:
        return image.copy(), mask.copy()
    # --------------------------------------------------
    # 5. Warp full image
    # --------------------------------------------------
    image_def = ndi.map_coordinates(image, coords, order=1, mode="nearest")

    # --------------------------------------------------
    # 6. Create new mask
    # --------------------------------------------------
    aug_mask = mask.copy()
    aug_mask[aug_mask == scar_label] = myo_label
    aug_mask[scar_def] = scar_label

    # --------------------------------------------------
    # 7. Smooth image transition near new scar
    # --------------------------------------------------
    boundary = ndi.gaussian_filter(scar_def.astype(np.float32), sigma=boundary_sigma)
    boundary = np.clip(boundary, 0.0, 1.0)

    blend_region = ndi.binary_dilation(scar | scar_def, iterations=roi_dilation)

    aug_image = image.copy()
    aug_image[blend_region] = image_def[blend_region] * boundary[blend_region] + image[blend_region] * (
        1.0 - boundary[blend_region]  # type: ignore
    )

    return aug_image, aug_mask


def remove_small_elements(segmentation_mask, min_size_remove=3):
    # Convert segmentation mask values greater than 0 to 1
    pred_mask = segmentation_mask > 0
    # Remove small objects (connected components) from the binary image
    mask = remove_small_objects(pred_mask, max_size=min_size_remove)
    # Multiply original segmentation mask with the mask to remove small objects
    clean_segmentation = segmentation_mask * mask
    return clean_segmentation


def make_volume(ndarray, voxel_spacing):
    volume = np.prod(voxel_spacing) * (ndarray.sum())
    return volume


@dataclass(frozen=True)
class BoundingBox:
    """An inclusive-exclusive bounding box in voxel coordinates (2D or 3D)."""

    mins: tuple[int, ...]
    maxs: tuple[int, ...]

    def slices(self) -> tuple[slice, ...]:
        return tuple(slice(a, b) for a, b in zip(self.mins, self.maxs))

    def apply(self, array: np.ndarray) -> np.ndarray:
        return array[self.slices()]


def crop_to_roi(image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    """Crop an array to a bounding box."""
    return bbox.apply(image)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a binary mask."""
    labelled, n = ndi.label(mask)
    if n <= 1:
        return mask
    sizes = ndi.sum(np.ones_like(labelled), labelled, index=range(1, n + 1))
    return labelled == (int(np.argmax(sizes)) + 1)


def _blood_pool_votes(
    vol: np.ndarray,
    core: np.ndarray,
    min_solidity: float,
    area_range: tuple[float, float],
) -> np.ndarray:
    """Count, per pixel, how many slices have a compact bright blob (blood pool) there."""
    n_pix = vol.shape[0] * vol.shape[1]
    a_min, a_max = area_range[0] * n_pix, area_range[1] * n_pix
    votes = np.zeros(vol.shape[:2], dtype=np.float32)
    for z in range(vol.shape[2]):
        s = ndi.gaussian_filter(vol[..., z], sigma=1.0)
        vals = s[core]
        if vals.size == 0 or vals.max() <= vals.min():
            continue
        bright = ndi.binary_opening(core & (s > filters.threshold_otsu(vals)), iterations=1)
        labelled, n = ndi.label(bright)
        keep = np.zeros(n + 1, dtype=bool)
        for region in measure.regionprops(labelled):
            keep[region.label] = a_min <= region.area <= a_max and region.solidity >= min_solidity
        votes += keep[labelled]
    return votes


def detect_roi(
    image: np.ndarray,
    spacing: tuple[float, float] | None = None,
    margin_mm: float = 50.0,
    margin_frac: float = 0.15,
    min_size: int = 200,
    max_crop_frac: float = 0.2,
    centre_sigma: float = 0.12,
    edge_frac: float = 0.1,
    min_solidity: float = 0.85,
    area_range: tuple[float, float] = (0.001, 0.15),
) -> BoundingBox:
    """Cardiac ROI grown outwards from the image centre until it covers the heart.

    1. Body core: Otsu on the slice-averaged image, fill holes, largest component,
       minus its outer ``edge_frac`` depth (chest wall / subcutaneous fat).
    2. Blood pools: on every slice, compact bright blobs in the body core vote for
       their pixels; votes are smoothed.
    3. Starting from the image centre, the heart is the voted region with the best
       score after a Gaussian prior on distance to the centre (``centre_sigma`` x size).
    4. The box stays centred on the image centre and opens outwards until it contains
       the heart region plus a margin (``margin_mm`` if ``spacing`` is given, else
       ``margin_frac`` of the image size); at least ``min_size`` pixels per side,
       and never removing more than ``max_crop_frac`` of the width or height.

    Args:
        image: 2D (H, W) or 3D (H, W, Z) image.
        spacing: In-plane pixel spacing in mm (e.g. ``nib_img.header.get_zooms()[:2]``).

    Returns:
        A BoundingBox. Falls back to the full image on failure.
    """
    img = image.astype(np.float32)
    full = BoundingBox((0,) * img.ndim, tuple(img.shape))
    vol = img[..., None] if img.ndim == 2 else img
    lo, hi = np.percentile(vol, (1, 99))
    if hi <= lo:
        return full
    vol = np.clip((vol - lo) / (hi - lo), 0, 1)
    shape = np.array(vol.shape[:2])
    centre = (shape - 1) / 2.0

    proj = ndi.gaussian_filter(vol.mean(axis=-1), sigma=2.0)
    body = proj > filters.threshold_otsu(proj)
    body = _largest_component(ndi.binary_fill_holes(ndi.binary_opening(body, iterations=2)))
    if not body.any():
        return full
    depth = ndi.distance_transform_edt(body)
    core = depth > edge_frac * depth.max()

    raw = _blood_pool_votes(vol, core, min_solidity, area_range)
    votes = ndi.gaussian_filter(raw, sigma=2.0)
    if votes.max() <= 0:
        return full
    rr, cc = np.indices(votes.shape)
    prior = np.exp(-((rr - centre[0]) ** 2 + (cc - centre[1]) ** 2) / (2 * (centre_sigma * shape.max()) ** 2))
    # Heart region: raw-vote component around the peak (smoothing would bridge nearby blobs).
    peak = np.unravel_index(np.argmax(np.where(raw > 0, votes * prior, 0)), votes.shape)
    labelled, _ = ndi.label(raw >= 0.25 * raw[peak])
    heart = np.argwhere(labelled == labelled[peak])

    # Open outwards from the image centre until the heart region (+ margin) fits.
    if spacing is not None:
        margin = margin_mm / np.asarray(spacing[:2], dtype=float)
    else:
        margin = margin_frac * shape
    half = np.abs(heart - centre).max(axis=0) + margin
    half = np.maximum(half, min_size / 2.0)
    half = np.maximum(half, (1.0 - max_crop_frac) * shape / 2.0)  # keep >= 80% of each side by default
    mins = np.maximum(np.floor(centre - half), 0).astype(int)
    maxs = np.minimum(np.ceil(centre + half) + 1, shape).astype(int)
    if img.ndim == 3:
        mins, maxs = list(mins) + [0], list(maxs) + [img.shape[2]]
    return BoundingBox(tuple(int(v) for v in mins), tuple(int(v) for v in maxs))


def crop_resize_image(
    image: np.ndarray,
    new_dim: int = 256,
    use_detect_roi: bool = False,
    spacing: tuple[float, float] | None = None,
    **roi_kwargs,
) -> tuple[np.ndarray, dict]:
    """Remove zero background, optionally crop to the cardiac ROI, square-crop and resize.

    Steps:
        1. Crop to the non-zero bounding box (removes zero padding).
        2. If ``use_detect_roi``: run :func:`detect_roi` on that crop and keep only the ROI.
        3. Make the crop square. Without ROI: centre crop to the shorter side.
           With ROI: grow the shorter side towards the longer one around the ROI centre,
           staying inside the non-zero region, so the heart is not cut.
        4. Resize in-plane to ``new_dim`` x ``new_dim``.

    All crops are combined into a single ``BoundingBox`` in original image coordinates.

    Parameters:
        image: Input image of shape (x, y, z).
        new_dim: Output in-plane size.
        use_detect_roi: Crop to the detected cardiac ROI after background removal.
        spacing: In-plane pixel spacing (mm), passed to :func:`detect_roi`.
        **roi_kwargs: Extra arguments for :func:`detect_roi` (e.g. ``margin_mm``, ``max_crop_frac``).

    Returns:
        (processed_image, restore_info)
            processed_image: shape (new_dim, new_dim, z')
            restore_info: {"bbox": BoundingBox, "original_shape": tuple, "new_dim": int}
    """
    # Step 1: Remove zero background
    nz = np.nonzero(image)
    if len(nz[0]) == 0:
        raise ValueError("Image is all zeros")
    lo = np.min(nz, axis=1)
    hi = np.max(nz, axis=1) + 1
    cropped = image[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))]

    # Step 2: Detect ROI inside the foreground (coordinates relative to `cropped`)
    if use_detect_roi:
        roi = detect_roi(cropped, spacing=spacing, **roi_kwargs)
        r_lo, r_hi = np.array(roi.mins[:2]), np.array(roi.maxs[:2])
        z_lo, z_hi = roi.mins[2], roi.maxs[2]
    else:
        r_lo, r_hi = np.zeros(2, dtype=int), np.array(cropped.shape[:2])
        z_lo, z_hi = 0, cropped.shape[2]

    # Step 3: Make square (in `cropped` coordinates)
    fg_shape = np.array(cropped.shape[:2])
    size = r_hi - r_lo
    if use_detect_roi:
        side = int(min(size.max(), fg_shape.min()))  # grow short side, but stay inside foreground
    else:
        side = int(size.min())  # centre crop to the short side
    start = r_lo + (size - side) // 2
    start = np.clip(start, 0, fg_shape - side)

    # Combine all crops into one bounding box in original image coordinates
    bbox = BoundingBox(
        (int(lo[0] + start[0]), int(lo[1] + start[1]), int(lo[2] + z_lo)),
        (int(lo[0] + start[0] + side), int(lo[1] + start[1] + side), int(lo[2] + z_hi)),
    )
    square = crop_to_roi(image, bbox)
    # print(f"Original image shape: {image.shape}, Cropped image shape: {square.shape}, BBox: {bbox}")
    # Step 4: Resize to new_dim
    if square.shape[0] != new_dim:
        resized = transform.resize(
            square,
            (new_dim, new_dim, square.shape[2]),
            anti_aliasing=True,
            preserve_range=True,
        ).astype(square.dtype)
    else:
        resized = square.copy()
    restore_info = {"bbox": bbox, "original_shape": image.shape, "new_dim": new_dim}
    return resized, restore_info


def crop_resize_mask(mask: np.ndarray, restore_info: dict) -> np.ndarray:
    """Crop a mask with the same bounding box as the image and resize it (nearest neighbour)."""
    if mask.shape != restore_info["original_shape"]:
        raise ValueError("Mask shape must match original image shape")
    square = crop_to_roi(mask, restore_info["bbox"])
    new_dim = restore_info["new_dim"]
    if square.shape[0] != new_dim:
        resized = transform.resize(
            square,
            (new_dim, new_dim, square.shape[2]),
            order=0,
            anti_aliasing=False,
        )
        return np.asarray(resized)
    return square.copy()


def restore_mask(processed_mask: np.ndarray, restore_info: dict) -> np.ndarray:
    """Resize a processed mask back to the bounding box size and paste it into the original shape."""
    bbox = restore_info["bbox"]
    box_shape = tuple(b - a for a, b in zip(bbox.mins, bbox.maxs))
    if processed_mask.shape != box_shape:
        resized = transform.resize(
            processed_mask,
            box_shape,
            order=0,
            anti_aliasing=False,
        )
        processed_mask = np.asarray(resized, dtype=processed_mask.dtype)
    restored = np.zeros(restore_info["original_shape"], dtype=np.int64)
    restored[bbox.slices()] = processed_mask.astype(np.int64)
    return restored


def preprocess_data_nii(image_path, dim_resize=256, cmr_type="cine"):
    data = {}
    image, data["affine"], data["header"] = load_nii(image_path)
    image = preprocess_cmr(image)

    resized_image, restore_info = crop_resize_image(image, dim_resize)
    data["restore_info"] = restore_info
    batch_images = []
    for i in range(resized_image.shape[-1]):
        slice_inputs = resized_image[..., i : i + 1]
        slices_image = torch.from_numpy(slice_inputs.transpose(-1, 0, 1))  # shape (1, dim_resize, dim_resize)
        batch_images.append(slices_image)

    batch_images = torch.stack(batch_images).float()  # shape (n,1, dim_resize, dim_resize)
    data["image"] = batch_images
    return data


def predict_patches(images, model, view, num_classes=4, batch_size=4, device="cuda", fp16=False):
    """return the patches"""
    probability_output = torch.zeros(
        (images.size(0), num_classes, images.size(2), images.size(3)),
        device=device,
    )

    batch_start = 0
    batch_end = batch_size
    while batch_start < images.size(0):
        image = images[batch_start:batch_end]
        with torch.inference_mode():
            image = image.to(device)
            if fp16:
                image = image.half()
            y_pred = model(image, view=view)
            probability_output[batch_start:batch_end] = y_pred
        batch_start += batch_size
        batch_end += batch_size
    return probability_output.cpu().numpy()


# shape (n, num_classes, dim_resize, dim_resize)


def load_nii(img_path):
    """Function to load a 'nii' or 'nii.gz' file."""
    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header  # type: ignore


def save_nii(img_path, data, affine, header):

    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)


from skimage.morphology import (
    label,
    remove_small_objects,
    remove_small_holes,
    closing,
    ball,
)


def postprocess_multiclass_volume(
    pred_volume,
    class_values=None,
    min_size=50,
    hole_area=50,
    smooth_radius=1,
    keep_largest=False,
):
    """
    Post-process a 3D multi-class segmentation volume.

    Parameters
    ----------
    pred_volume : np.ndarray
        3D predicted mask with shape [H, W, Z] or [Z, H, W].
        Each voxel has integer class label.

    class_values : list[int] or None
        Classes to process. If None, process all labels > 0.

    min_size : int
        Remove connected components smaller than this.

    hole_area : int
        Fill holes smaller than this.

    smooth_radius : int
        Radius for 3D morphological closing.

    keep_largest : bool
        If True, keep only the largest connected component for each class.

    Returns
    -------
    processed_volume : np.ndarray
        Post-processed segmentation volume.
    """

    pred_volume = np.asarray(pred_volume)

    if pred_volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {pred_volume.shape}")

    if class_values is None:
        class_values = [c for c in np.unique(pred_volume) if c > 0]

    processed_volume = np.zeros_like(pred_volume)

    for class_value in class_values:
        mask = pred_volume == class_value

        if mask.sum() == 0:
            continue

        # Remove tiny components
        mask = remove_small_objects(mask, max_size=min_size)

        if mask.sum() == 0:
            continue

        # Keep only largest connected component
        if keep_largest:
            labeled = label(mask)
            component_ids, counts = np.unique(labeled, return_counts=True)

            # remove background label 0
            component_ids = component_ids[component_ids != 0]
            counts = counts[1:]

            if len(component_ids) == 0:
                continue

            largest_component = component_ids[np.argmax(counts)]
            mask = labeled == largest_component

        # Fill small holes
        mask = remove_small_holes(mask, max_size=hole_area)

        # Smooth in 3D
        if smooth_radius > 0:
            mask = closing(mask, ball(smooth_radius))

        processed_volume[mask] = class_value

    return processed_volume


from scipy.ndimage import label, binary_dilation, generate_binary_structure


def postprocess_scar_between_lv_myo_3d(
    pred,
    prob=None,
    lv_label=1,
    myo_label=2,
    scar_label=3,
    min_ratio=0.03,
    max_ratio=0.4,
    min_scar_voxels=50,
    min_component_voxels=5,
    myo_dilation_iter=5,
    lv_dilation_iter=5,
    max_reduce_fraction=0.1,
    require_between_lv_myo=False,
):
    """
    Conservative 3D scar post-processing for CMR.

    Anatomical assumption:
        scar should be between / close to both LV cavity and LV myocardium.

    Removed scar is converted to myocardium, not background.

    Labels:
        1 = LV cavity
        2 = LV myocardium
        3 = myocardial scar
        4 = RV cavity
    """

    out = pred.copy()

    lv = out == lv_label
    myo = out == myo_label
    scar = out == scar_label

    scar_voxels = int(scar.sum())
    myo_total = myo | scar
    myo_total_voxels = int(myo_total.sum())

    if scar_voxels == 0:
        return out

    if myo_total_voxels == 0:
        out[scar] = myo_label
        return out

    scar_ratio = scar_voxels / myo_total_voxels

    # --------------------------------------------------
    # Case 1: tiny false-positive scar
    # --------------------------------------------------
    if scar_voxels < min_scar_voxels or scar_ratio < min_ratio:
        out[scar] = myo_label
        return out

    # --------------------------------------------------
    # Step 1: anatomical validity
    # Scar should be near both LV cavity and myocardium
    # --------------------------------------------------
    lv_region = binary_dilation(lv, iterations=lv_dilation_iter)
    myo_region = binary_dilation(myo, iterations=myo_dilation_iter)

    if require_between_lv_myo:
        valid_scar = scar & lv_region & myo_region
    else:
        valid_scar = scar & (lv_region | myo_region)

    # Safety: if this removes too much, keep original scar
    # This protects true scar patients.
    if valid_scar.sum() < 0.5 * scar_voxels:
        valid_scar = scar

    # --------------------------------------------------
    # Step 2: remove tiny 3D scar components
    # --------------------------------------------------
    structure = generate_binary_structure(3, 2)
    cc, num = label(valid_scar, structure=structure)

    cleaned_scar = np.zeros_like(valid_scar, dtype=bool)

    for i in range(1, num + 1):
        comp = cc == i
        comp_size = int(comp.sum())

        if comp_size >= min_component_voxels:
            cleaned_scar |= comp

    # Safety: if component filtering removes too much, keep original scar
    if cleaned_scar.sum() < 0.5 * scar_voxels:
        cleaned_scar = scar

    cleaned_ratio = cleaned_scar.sum() / myo_total_voxels

    # --------------------------------------------------
    # Step 3: only reduce if scar is extremely too much
    # --------------------------------------------------
    if cleaned_ratio > max_ratio and prob is not None:
        target_keep_voxels = int(max_ratio * myo_total_voxels)

        # Do not remove more than max_reduce_fraction of current scar
        min_keep_voxels = int((1.0 - max_reduce_fraction) * cleaned_scar.sum())
        target_keep_voxels = max(target_keep_voxels, min_keep_voxels)

        # Extract scar probability
        if prob.shape[0] > scar_label and prob.shape[1:] == pred.shape:
            scar_prob = prob[scar_label]
        elif prob.shape[-1] > scar_label and prob.shape[:-1] == pred.shape:
            scar_prob = prob[..., scar_label]
        else:
            raise ValueError("prob shape must be [C, ...] or [..., C] matching pred")

        candidate_idx = np.argwhere(cleaned_scar)
        candidate_probs = scar_prob[cleaned_scar]

        order = np.argsort(candidate_probs)[::-1]
        keep_idx = candidate_idx[order[:target_keep_voxels]]

        reduced_scar = np.zeros_like(cleaned_scar, dtype=bool)
        reduced_scar[tuple(keep_idx.T)] = True
        cleaned_scar = reduced_scar

    # --------------------------------------------------
    # Final update:
    # removed scar -> myocardium
    # kept scar remains scar
    # --------------------------------------------------
    out[out == scar_label] = myo_label
    out[cleaned_scar] = scar_label

    return out


# new_pred = postprocess_multiclass_volume(pred, min_size=500, hole_area=50, smooth_radius=1, keep_largest=True)

# print(f"Dice Score: {dice_score(new_pred, mask, class_id=1)}")
# print(f"Dice Score: {dice_score(new_pred, mask, class_id=2)}")
# print(f"Dice Score: {dice_score(new_pred, mask, class_id=3)}")
