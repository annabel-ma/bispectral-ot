import torch
import os
import random as pyrandom 
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from typing import Optional, List, Dict
from PIL import Image
from torch.utils.data import Dataset

from bispectrum import * 
from angular_bispectrum import *

SEED = 12345

os.environ["PYTHONHASHSEED"] = str(SEED)
pyrandom.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

print(f"[utils] Global seed set to {SEED} for reproducibility.")

def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    pyrandom.seed(worker_seed)

def make_generator():
    g = torch.Generator()
    g.manual_seed(SEED)
    return g

def build_transform(tensor=True, normalization=None, resize=None):
    ops = []
    if resize is not None:
        ops.append(transforms.Resize((resize, resize), interpolation=InterpolationMode.BILINEAR))
    if tensor:
        ops.append(transforms.ToTensor())
    if normalization is not None: 
        ops.append(transforms.Normalize(*normalization))
    return transforms.Compose(ops)

# transforms for images (mean and std that i have precomp) 
MNIST_NORM = ((0.13066047871907552,), (0.3081078052524796,))
FASHION_NORM = ((0.28604060169855755,), (0.35302425250395036,))
KMNIST_NORM = ((0.19176215070088704,), (0.3483428300416729,))
USPS_NORM = ((0.24687695794268377,), (0.29887581237380284,))
EMNIST_NORM = ((0.1722273071606954,), (0.33094662784564527, ))

COIL100_MEAN = torch.tensor([0.3073, 0.2593, 0.2063])
COIL100_STD = torch.tensor([0.2691, 0.2178, 0.1962])

# for wilds (loading with a dataloader)

def is_near_black_or_white(img, threshold=0.05):
    min_val = img.min()
    max_val = img.max()

    return min_val >= (1 - threshold) or max_val <= threshold

def extract_small_data_loader(data_loader, N=200):
    extracted = []
    class_counts = {}
    n_classes = 2 #dataset.n_classes

    # Adjust based on batch structure
    for batch in data_loader:
        inputs, labels = batch[0], batch[1]

        for img, label in zip(inputs, labels):
            label = label.item()
            if label not in class_counts:
                class_counts[label] = 0
            if class_counts[label] < N and not is_near_black_or_white(img):
                extracted.append((img, label))
                class_counts[label] += 1
            # Check if we have collected N samples for all classes
            if len(class_counts) == n_classes and all(count >= N for count in class_counts.values()):
                return extracted
    return extracted

def extract_two_disjoint_sets_data_loader(data_loader, N=200):
    extracted1 = []
    extracted2 = []
    class_counts1 = {}
    class_counts2 = {}
    n_classes = 2

    for c in range(n_classes):
        class_counts1[c] = 0
        class_counts2[c] = 0

    for batch in data_loader:
        inputs, labels = batch[0], batch[1]

        for img, label in zip(inputs, labels):
            label = label.item()

            if class_counts1[label] < N and not is_near_black_or_white(img):
                extracted1.append((img, label))
                class_counts1[label] += 1
            elif class_counts2[label] < N and not is_near_black_or_white(img):
                extracted2.append((img, label))
                class_counts2[label] += 1

            if (all(class_counts1[c] >= N for c in range(n_classes)) and
                all(class_counts2[c] >= N for c in range(n_classes))):
                return extracted1, extracted2

    return extracted1, extracted2

# sans data loader

def _label_tensor(ds: torch.utils.data.Dataset) -> torch.Tensor:
    for attr in ("targets", "labels"):
        if hasattr(ds, attr):
            y = getattr(ds, attr)
            return y if isinstance(y, torch.Tensor) else torch.as_tensor(y)
    return torch.as_tensor([int(ds[i][1]) for i in range(len(ds))]) 

def _ensure_chw(x: torch.Tensor) -> torch.Tensor:
    return x if x.ndim == 3 else x.unsqueeze(0)

def rotate_degree(image_tensor: torch.Tensor, angle: float) -> torch.Tensor:
    x = _ensure_chw(image_tensor)

    _, h, w = x.shape
    center = (w / 2.0, h / 2.0)
    
    return TF.rotate(
        x, angle,
        interpolation=InterpolationMode.BILINEAR,
        center=center,
        expand=False
    )

def rotate_random_degree(image_tensor: torch.Tensor) -> torch.Tensor:
    x = _ensure_chw(image_tensor)
    _, h, w = x.shape
    center = (w / 2.0, h / 2.0)
    angle = pyrandom.uniform(0.0, 360.0)    
    return TF.rotate(
        x, angle,
        interpolation=InterpolationMode.BILINEAR,
        center=center,
        expand=False
    )

def extract_disjoint_sets(ds, N: Optional[int] = None, seed: Optional[int] = None, contiguous = True, rotate = False, unbalanced = False, num_small=1):
    if hasattr(ds, "labels"):
        y = ds.labels
    else: 
        y = _label_tensor(ds)
        
    idx_by_class: Dict[int, List[int]] = {}
    for i, c in enumerate(y.tolist()):
        idx_by_class.setdefault(int(c), []).append(i)

    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        if contiguous:
            for c, idxs in idx_by_class.items():
                if gen is not None and len(idxs) > 0:
                    start = torch.randint(low=0, high=len(idxs), size=(1,), generator=gen).item()
                    idx_by_class[c] = idxs[start:] + idxs[:start]
        else: 
            for c, idxs in idx_by_class.items():
                perm = torch.randperm(len(idxs), generator=gen).tolist()
                idx_by_class[c] = [idxs[i] for i in perm]

    counts = {c: len(v) for c, v in idx_by_class.items()}
    if N is None:
        n_per_class = {c: counts[c] // 2 for c in counts}
    else:
        shortages = {c: counts[c] for c in counts if counts[c] < 2 * N}
        if shortages:
            raise ValueError(
                f"Not enough to allocate two sets with N={N} per class. "
                f"Each class needs ≥ {2*N}. Shortages: {shortages}"
            )
        n_per_class = {c: int(N) for c in counts}

    setA, setB = [], []
    for c, idxs in idx_by_class.items():
        if unbalanced:
            A_indices = idxs[num_small:]
            B_indices = idxs[:num_small]
        else: 
            n_c = n_per_class[c]
            if n_c <= 0:
                continue
            A_indices = idxs[:n_c]
            B_indices = idxs[n_c:2*n_c]

        if rotate:
            for i in A_indices:
                x, lab = ds[i]
                xr = rotate_random_degree(x)
                setA.append((xr, int(lab)))
        else:
            for i in A_indices:
                x, lab = ds[i] 
                x = _ensure_chw(x)
                setA.append((x, int(lab)))

        for i in B_indices:
            x, lab = ds[i] 
            x = _ensure_chw(x)
            setB.append((x, int(lab)))

    return setA, setB, n_per_class

def prepare_bispec(data, num_angles = 40, bispec_type = 'angular'):
    N = len(data)

    if bispec_type == '2D': 
        return {
        'ys': [data[i][1] for i in range(N)],
        'xs': [_ensure_chw(data[i][0]) for i in range(N)],
        'bs': [torch.stack([bispectrum_2d(data[i][0]).real, bispectrum_2d(data[i][0]).imag]) for i in range(N)]
        }
        
    bis = PolarBispec(num_angles)  

    return {
        'ys': [data[i][1] for i in range(N)],
        'xs': [_ensure_chw(data[i][0]) for i in range(N)],
        'bs': [torch.stack([bis.bispec(data[i][0]).real, bis.bispec(data[i][0]).imag]) for i in range(N)]
    }
