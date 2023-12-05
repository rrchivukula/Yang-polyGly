from __future__ import print_function, unicode_literals, absolute_import, division

import sys
import os
import contextlib
import argparse
from glob import glob
import pickle
from tqdm import tqdm as tqdm_

import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["image.interpolation"] = "none"
matplotlib.rcParams["font.family"] = "Helvetica"
matplotlib.rcParams["font.size"] = 8
matplotlib.rcParams["axes.facecolor"] = (1, 1, 1, 0)

from aicsimageio import AICSImage
from tifffile import imread, imwrite
import imageio
from csbdeep.utils import normalize
from stardist import random_label_cmap
from stardist.models import StarDist3D

from sklearn.mixture import GaussianMixture
from skimage.measure import regionprops_table
from skimage.segmentation import active_contour
from skimage.transform import rescale
from skimage.filters import gaussian

np.random.seed(6)
lbl_cmap = random_label_cmap()


def tqdm(iterable, **kwargs):
    pbar = tqdm_(
        iterable, bar_format="{desc:<20}{percentage:3.0f}%|{bar:50}{r_bar}", **kwargs
    )
    for x in pbar:
        if type(x) is str:
            pbar.set_postfix_str(x)
        yield x


def load_images(paths):
    """
    from list of paths to image files, return 3 lists:
        name of ROI
        image data for ROI
        (file, scene) for ROI
    where each file may have more than one ROI
    """
    names, data, files = [], [], []
    # print(paths)
    images = list(map(AICSImage, paths))
    for path, img in zip(tqdm(paths, desc="loading images"), images):
        # print(path, img)
        for i, scene in enumerate(tqdm(img.scenes, desc="loading scenes")):
            # print(scene)
            img.set_scene(scene)
            names.append(f"{os.path.splitext(os.path.basename(path))[0]}_{i}")
            data.append(img.get_image_data("CZYX", T=0))
            files.append((path, scene))
    return names, data, files


def get_root(F):
    """
    get root folder for a list of files F
    """
    if len(set([f[0] for f in F])) == 1:
        return os.path.dirname(F[0][0])
    return os.path.commonpath([f[0] for f in F])


def load_masks(N, F, folder):
    """
    load masks from specified folder
    """
    root = get_root(F)
    labels = [
        imread(os.path.join(root, folder, f"{name}.tif"))
        for name in tqdm(N, desc=f"reading {folder:.12}")
    ]
    return labels


def save_masks(N, Y, F, folder):
    """
    save masks Y to specified folder
    """
    root = get_root(F)
    if not os.path.exists(os.path.join(root, folder)):
        os.mkdir(os.path.join(root, folder))
    for name, pred in zip(tqdm(N, desc=f"writing {folder:.12}"), Y):
        imwrite(os.path.join(root, folder, f"{name}.tif"), pred, compression="deflate")


def predict_nuclei(N, X, F, args):
    """
    using StarDist and StarDist OPP, predict nuclear masks and write to "masks_nucleus" folder
    """
    X0 = [x[0, :, :, :] for x in X]
    axis_norm = (0, 1, 2)
    X0 = [
        normalize(x, 1, 99.8, axis=axis_norm)
        for x in tqdm(X0, desc="normalizing nuclei")
    ]

    labels = []
    model = StarDist3D(None, name=args.nuc_model, basedir="models")
    for x in tqdm(X0, desc="segmenting nuclei"):
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            labels.append(
                model.predict_instances(
                    x,
                    n_tiles=model._guess_n_tiles(x),
                    show_tile_progress=False,
                    return_predict=False,
                )[0]
            )

    save_masks(N, labels, F, "masks_nucleus")


def show_predicted_nuclei(N, X, F):
    """
    plot nuclei and nuclear masks
    """
    X0 = [x[0, :, :, :] for x in X]
    axis_norm = (0, 1, 2)
    X0 = [
        normalize(x, 1, 99.8, axis=axis_norm)
        for x in tqdm(X0, desc="normalizing nuclei")
    ]
    labels = load_masks(N, F, "masks_nucleus")

    fig, ax = plt.subplots(2, 3, figsize=(9, 6))
    for i, (ax_x, ax_l, x, l) in enumerate(zip(ax[0], ax[1], X0[:3], labels[:3])):
        ax_x.imshow(np.amax(x, axis=0), clim=(0, 2), cmap="gray")
        ax_x.set_title(i)
        ax_x.axis("off")
        ax_l.imshow(np.amax(l, axis=0), cmap=lbl_cmap)
        ax_l.axis("off")
    plt.tight_layout()
    plt.show()


def contour(img, bounds, n_points=10):
    """
    convert segmented 2D pixel mask into contour

    Parameters
    ----------
    img : np.ndarray
        2D pixel mask
    bounds : tuple
        left, right, top, bottom coordinates to initialize active contour
    n_points : int
        number of contour points per side
    """
    scale_factor = 4
    rescaled = rescale(img, scale_factor, anti_aliasing=False)
    blur = gaussian(rescaled, sigma=3, preserve_range=True)
    bounds = np.array(bounds)
    l, r, t, b = bounds * scale_factor
    x = np.concatenate(
        [
            np.linspace(l, r, num=n_points),
            np.linspace(r, r, num=n_points),
            np.linspace(r, l, num=n_points),
            np.linspace(l, l, num=n_points),
        ]
    )
    y = np.concatenate(
        [
            np.linspace(t, t, num=n_points),
            np.linspace(t, b, num=n_points),
            np.linspace(b, b, num=n_points),
            np.linspace(b, t, num=n_points),
        ]
    )
    init = np.array([y, x]).T
    contour = active_contour(blur, init, boundary_condition="periodic")
    return contour / scale_factor - 0.5, init / scale_factor - 0.5


def composite_image(
    channels,
    cmaps=np.array([[0, 0, 1], [0, 1, 0], [1, 0, 1]]),
    clims=[[100, 4096]] * 3,
    ax=None,
):
    """
    combine image channels into composite image

    Parameters
    ----------
    channels :
        channel images
    cmaps :
        color map for each channel (RGB tuple)
    clims :
        min and max for each channel
    ax :
        axes to show composite image
    """
    chs = []
    for ch, cm, cl in zip(channels, cmaps, clims):
        if ch is None:
            continue
        a, b = cl
        if a is None:
            a = ch.min()
        if b is None:
            b = ch.max()
        x = np.moveaxis(np.tile(ch, (3, 1, 1)), 0, 2) * cm
        x = (x - a) / (b - a)
        chs.append(x)
    if ax is None:
        plt.imshow(np.clip(np.sum(chs, axis=0), 0, 1))
    else:
        ax.imshow(np.clip(np.sum(chs, axis=0), 0, 1))


def analyze_nuclei(N, X, F, args):
    """
    analyze segmented nuclei for fam98b intensity and aggregate classification
    """
    root = get_root(F)

    # regionprops intensity metrics
    def quartiles(regionmask, intensity):
        """compute 25th, 50th, and 75th percentile intensities"""
        return np.percentile(intensity[regionmask], q=(25, 50, 75))

    def gmm2(regionmask, intensity_a, intensity_b):
        """fit 2-component gaussian mixture model to intensity_a to classify
        pixels into a-low and a-high; return median intensity_b of a-low
        pixels"""
        if regionmask.sum() < 100:
            return np.nan
        gm = GaussianMixture(n_components=2).fit(
            intensity_a[regionmask][..., np.newaxis]
        )
        label_min = np.argmin(gm.means_[:, 0])
        predictions = gm.predict(intensity_a[regionmask][..., np.newaxis])
        return np.median(intensity_b[regionmask][predictions == label_min])

    def analyze_roi(index, x, mask_nuc):
        """calculate bbox and intensity metrics for each nuclei in ROI"""
        df = pd.DataFrame(
            regionprops_table(
                mask_nuc, properties=["label", "bbox", "area", "centroid"]
            )
        ).set_index("label")
        df = df.join(
            pd.DataFrame(
                regionprops_table(
                    mask_nuc,
                    x[2, :, :, :],
                    properties=["label"],
                    extra_properties=[quartiles],
                )
            ).set_index("label")
        )
        df = df.join(
            pd.DataFrame(
                regionprops_table(
                    mask_nuc,
                    x[1, :, :, :],
                    properties=["label"],
                    extra_properties=[quartiles],
                )
            ).set_index("label"),
            rsuffix="_ch1",
        )
        df = df.join(
            df.index.to_series()
            .map(lambda i: gmm2(mask_nuc == i, x[1, :, :, :], x[2, :, :, :]))
            .rename("gmm2")
        )
        # background value is mean non-nuclear fam98b signal
        df["background"] = x[2, :, :, :].flatten()[~mask_nuc.flatten()].mean()
        df = df.reset_index()
        df.insert(loc=0, column="roi", value=index)
        return df

    labels = load_masks(N, F, "masks_nucleus")
    if os.path.exists(os.path.join(root, "analyzed.pkl")):
        with open(os.path.join(root, "analyzed.pkl"), "rb") as f:
            data = pickle.load(f)
    else:
        data = pd.concat(
            list(
                map(analyze_roi, tqdm(range(len(N)), desc="analyzing ROIs"), X, labels)
            )
        ).reset_index(drop=True)
        with open(os.path.join(root, "analyzed.pkl"), "wb") as f:
            pickle.dump(data, f)

    def crop_images(df, size):
        """pre-process images for aggregate classification: crop, mask, and
        z-project image around each nucleus and convert to 8-bit 3-channel
        image; concatenate all images into array"""
        images = []
        # cache padded arrays for each ROI to avoid re-computation (nuclei from
        # same ROI are contiguous in df)
        cache_index, cache_padded_image, cache_padded_label = None, None, None
        for i, row in tqdm(df.iterrows(), desc="cropping nuclei", total=len(df)):
            center = row[["centroid-0", "centroid-1", "centroid-2"]].apply(int).values
            bounds = (
                slice(int(row["bbox-0"]) + size, int(row["bbox-3"]) + size),
                slice(center[1] - size // 2 + size, center[1] + size // 2 + size),
                slice(center[2] - size // 2 + size, center[2] + size // 2 + size),
            )
            roi_index = int(row["roi"])
            if roi_index == cache_index:
                padded_image = cache_padded_image
                padded_label = cache_padded_label
            else:
                padded_image = [
                    np.pad(X[roi_index][i, :, :, :], size) for i in range(3)
                ]
                cache_padded_image = padded_image
                padded_label = np.pad(labels[roi_index], size)
                cache_padded_label = padded_label
                cache_index = roi_index

            cropped_image = [ch[bounds] for ch in padded_image]
            cropped_mask = (padded_label == row["label"])[bounds]

            # for classification, mask out signal outside of segmented nucleus
            proj = [np.amax(ch * cropped_mask, axis=0) for ch in cropped_image]
            # 16 converts from 12-bit to 8-bit
            composite = (np.stack(proj, axis=-1) / 16).astype(np.uint8)
            images.append(composite)
        return np.stack(images)

    def classify_nuclei(df, size=80, threshold=0.5):
        """use trained aggregate model to classify nuclei"""
        if os.path.exists(os.path.join(root, "cropped.pkl")):
            with open(os.path.join(root, "cropped.pkl"), "rb") as f:
                images = pickle.load(f)
        else:
            images = crop_images(df, size=size)
            with open(os.path.join(root, "cropped.pkl"), "wb") as f:
                pickle.dump(images, f)

        model = tf.keras.models.load_model("./models/aggregate2.keras")
        scores = model.predict(images)[:, 0]
        return (
            pd.DataFrame(
                {"score": scores, "aggregate": scores < threshold}, index=df.index
            ),
            images,
        )

    if args.classify_aggregate:
        pred, cropped_images = classify_nuclei(data, size=80, threshold=0.5)
        data = data.join(pred)
    else:
        data["score"] = None
        data["aggregate"] = None

    def show_cell(
        row, axs, size=80, outline=True, merge=True, channels=range(3), **kwargs
    ):
        """for a given nucleus, show individual channels and/or merge"""
        center = row[["centroid-0", "centroid-1", "centroid-2"]].apply(int).values
        bounds = (
            slice(int(row["bbox-0"]) + size, int(row["bbox-3"]) + size),
            slice(center[1] - size // 2 + size, center[1] + size // 2 + size),
            slice(center[2] - size // 2 + size, center[2] + size // 2 + size),
        )
        roi_index = int(row["roi"])
        padded_image = [np.pad(X[roi_index][i, :, :, :], size) for i in range(3)]
        padded_label = np.pad(labels[roi_index], size)
        cropped_image = [ch[bounds] for ch in padded_image]
        cropped_mask = (padded_label == row["label"])[bounds]

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
            for x in ("left", "right", "top", "bottom"):
                ax.spines[x].set_visible(False)

        for i, im in zip(channels, cropped_image):
            x = [None] * len(cropped_image)
            x[i] = np.amax(im, axis=0)
            composite_image(x, ax=axs[i], **kwargs)
        if merge:
            composite_image(
                [np.amax(x, axis=0) for x in cropped_image], ax=axs[-1], **kwargs
            )

        if outline:
            nuc_mip = np.amax(cropped_mask, axis=0)
            snake_init = (
                row["bbox-2"] - bounds[2].start - 2 + size,
                row["bbox-5"] - bounds[2].start + 2 + size,
                row["bbox-1"] - bounds[1].start - 2 + size,
                row["bbox-4"] - bounds[1].start + 2 + size,
            )
            snake, _ = contour(nuc_mip, snake_init, n_points=50)
            for ax in axs:
                ax.plot(snake[:, 1], snake[:, 0], ls="dotted", lw=0.5, c="w")

    if not os.path.exists(os.path.join(root, "results")):
        os.mkdir(os.path.join(root, "results"))

    if args.examples:
        # for predicted aggregate-containing and -lacking nuclei, randomly sample
        # 15 nuclei and display in figure
        for ag in (True, False):
            sample = data[data["aggregate"] == ag].sample(
                min(15, len(data[data["aggregate"] == ag]))
            )
            if len(sample) == 0:
                print(f"no nuclei with aggregate {ag}.")
                continue

            fig, axs = plt.subplots(len(sample), 4, figsize=(4, len(sample)))
            if len(axs.shape) == 1:
                axs = [axs]
            for i, (idx, row) in enumerate(sample.iterrows()):
                show_cell(
                    row,
                    axs[i],
                    size=80,
                    channels=[0, 1, 2],
                    outline=True,
                    clims=[
                        [200, None],
                        [300, 2000],
                        [50, 2500],
                    ],
                )
            for ax, t, c in zip(
                axs[0],
                ["Hoechst", "ubiquitin", "FAM98B", "merge"],
                ["blue", "green", "magenta", "black"],
            ):
                ax.set_title(t, fontdict={"weight": "bold", "color": c})
            plt.subplots_adjust(wspace=0.025, hspace=0.025)
            plt.tight_layout()
            plt.savefig(os.path.join(root, "results", f"examples_{ag}.svg"))

    data.to_excel(os.path.join(root, "results", "data.xlsx"))

    if args.qc_images:
        if not os.path.exists(os.path.join(root, "qc_images")):
            os.mkdir(os.path.join(root, "qc_images"))

        for i, (idx, row) in tqdm(
            enumerate(data[data["aggregate"]].iterrows()),
            total=len(data[data["aggregate"]]),
            desc="writing qc images",
        ):
            fig, axs = plt.subplots(1, 4, figsize=(4, 1))
            show_cell(
                row,
                axs,
                size=80,
                channels=[0, 1, 2],
                outline=True,
                clims=[
                    [200, None],
                    [300, 2000],
                    [50, 2500],
                ],
            )
            for ax, t, c in zip(
                axs,
                ["Hoechst", "ubiquitin", "FAM98B", "merge"],
                ["blue", "green", "magenta", "black"],
            ):
                ax.set_title(t, fontdict={"weight": "bold", "color": c})
            plt.subplots_adjust(wspace=0.025, hspace=0.025)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    root,
                    "qc_images",
                    "roi{}_label{}.png".format(row["roi"], row["label"]),
                ),
                dpi=300,
            )
            plt.close()

        data[data["aggregate"]][["roi", "label", "aggregate"]].to_excel(
            os.path.join(root, "qc_images", "curated.xlsx")
        )

    if args.training_images:
        if not os.path.exists(os.path.join(root, "training_images")):
            os.mkdir(os.path.join(root, "training_images"))
            os.mkdir(os.path.join(root, "training_images", "True"))
            os.mkdir(os.path.join(root, "training_images", "False"))

        copy = data.reset_index(drop=True)
        for ag in (True, False):
            sample = copy[copy["aggregate"] == ag].sample(
                min(20, len(copy[data["aggregate"] == ag]))
            )
            for idx, row in sample.iterrows():
                img = cropped_images[idx]
                imageio.imwrite(
                    os.path.join(root, "training_images", str(ag), f"{idx}.png"), img
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="file(s)")
    parser.add_argument(
        "--nuc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="segment nuclei",
    )
    parser.add_argument(
        "--nuc-model",
        action="store",
        default="nucleus2",
        type=str,
        help="nuclear segmentation model",
    )
    parser.add_argument(
        "--show-nuc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="segment nuclei",
    )
    parser.add_argument(
        "--analyze",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="analyze nuclei",
    )
    parser.add_argument(
        "--classify-aggregate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="classify nuclei as having or not having aggregates",
    )
    parser.add_argument(
        "--examples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="export example images",
    )
    parser.add_argument(
        "--qc-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="export qc images",
    )
    parser.add_argument(
        "--training-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="export cropped training images",
    )
    args = parser.parse_args()

    N, X, F = load_images(args.files)
    if args.nuc:
        predict_nuclei(N, X, F, args)
    if args.show_nuc:
        show_predicted_nuclei(N, X, F)
    if args.analyze:
        analyze_nuclei(N, X, F, args)
