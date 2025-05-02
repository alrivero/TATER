import os
from pathlib import Path
from typing import List, Union, Sequence, Tuple
import re

import torch
import matplotlib.pyplot as plt


def save_attention_maps(
    attn_data: Sequence[Tuple[str, torch.Tensor]],
    save_dir: Union[str, os.PathLike] = "attn_vis",
    batch_idx: int = 0,
    vmax: float = None,
    cmap: str = "viridis",
    dpi: int = 150,
    show_progress: bool = True
) -> Sequence[Path]:
    """
    Save each (tag, layer-head) attention slice as a PNG.

    Parameters
    ----------
    attn_data : list of (tag, Tensor)
        Each tensor has shape (B, H, tgt_len, src_len).
    save_dir : path-like
        Directory where .png files will be written (created if missing).
    batch_idx : int, default=0
        Which sample in the batch to visualize.
    vmax : float or None
        Upper bound for color scaling.  If None we pick the max of each map.
    cmap : str
        Any valid matplotlib colormap.
    dpi : int
        Resolution of saved images.
    show_progress : bool
        Print progress per tag.

    Returns
    -------
    List[Path]
        Paths of all images that were saved.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize(tag: str) -> str:
        # replace non-alphanumeric with underscore
        return re.sub(r"[^\w\-]+", "_", tag)

    saved_paths = []
    for tag, A in attn_data:
        A = A.detach().cpu()           # (B, heads, tgt, src)
        B, H, T, S = A.shape
        safe_tag = _sanitize(tag)

        for head_id in range(H):
            attn_map = A[batch_idx, head_id]  # (tgt, src)
            vmax_use = float(attn_map.max() if vmax is None else vmax)

            fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)
            im = ax.imshow(
                attn_map,
                vmin=0.0, vmax=vmax_use,
                cmap=cmap,
                aspect="auto"
            )
            ax.set_xlabel("Key / Source position")
            ax.set_ylabel("Query / Target position")
            ax.set_title(f"{tag}  Head {head_id}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            fname = save_dir / f"{safe_tag}_head{head_id:02d}.png"
            fig.savefig(fname, bbox_inches="tight")
            plt.close(fig)

            saved_paths.append(fname)

        if show_progress:
            print(f"[save_attention_maps] {tag!r}: {H} heads → {H} images")

    return saved_paths


# ────────────────────────────────────────────────────────────────────────────
#  VISUALISE OR SAVE EVERY ATTENTION MAP THAT THE MODEL RETURNS
# ────────────────────────────────────────────────────────────────────────────
import os
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np     # only for safe type-checking / conversion

def visualise_attention(
    records: List[Tuple[str, "torch.Tensor"]],
    batch_idx: int = 0,
    head_idx:  int = 0,
    max_plots: int = 12,
    save_dir:  Union[str, os.PathLike, None] = None,
    show: bool = True,
    dpi:  int = 150,
):
    """
    Parameters
    ----------
    records     list of (tag:str, tensor[B,H,T,S])  – output of the model
    batch_idx   which batch element to pick
    head_idx    which head to show for every map
    max_plots   upper bound on how many figures to create (one per attention map)
    save_dir    folder to write PNGs to; if None → do not save
    show        call plt.show() so they pop up in an interactive session
    dpi         resolution for the saved/interactive figure
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    plotted = 0
    for tag, A in records:
        if plotted >= max_plots:
            break
        # -------------------- tensor hygiene -------------------- #
        if hasattr(A, "detach"):           # torch Tensor
            A = A.detach().cpu().numpy()
        elif not isinstance(A, np.ndarray):
            raise TypeError(f"Expected torch.Tensor or ndarray, got {type(A)}")

        if A.ndim != 4:
            continue                       # silently skip malformed entries

        attn = A[batch_idx, head_idx]      # shape (tgt_len, src_len)

        # -------------------- draw ------------------------------ #
        plt.figure(figsize=(4, 4), dpi=dpi)
        plt.imshow(attn, aspect="auto")    # default colormap, no styling
        plt.title(f"{tag}   head {head_idx}")
        plt.xlabel("Key / Source position")
        plt.ylabel("Query / Target position")
        plt.colorbar()

        if save_dir is not None:
            fname = save_dir / f"{tag}_head{head_idx:02d}.png"
            plt.savefig(fname, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()

        plotted += 1