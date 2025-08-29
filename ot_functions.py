import pandas as pd
import numpy as np
import os, time, errno, pickle, tempfile, hashlib
import seaborn as sns
import matplotlib.pyplot as plt
import ast 
import ot
import torch
import torchvision.transforms as transforms
import time
from typing import Optional
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path
import scipy.spatial.distance as ssd


def computeOT(data1, data2, feature = 'xs',  method = 'sinkhorn_epsilon_scaling', reg = 0.1, num_itermax = 100000, is_verbose = False, p=2, dist_type = None, is_sq = True):
    N = len(data1['ys'])
    L = len(data2['ys'])

    if dist_type is not None: 
        M = ssd.cdist(torch.stack(data1[feature]).reshape(N, -1), torch.stack(data2[feature]).reshape(L, -1), metric=dist_type)
        M = torch.tensor(M)
    else:
        if p == 2:
            if is_sq:
                M = torch.cdist(torch.stack(data1[feature]).reshape(N, -1), torch.stack(data2[feature]).reshape(L, -1), p=p) ** 2
            else:
                M = torch.cdist(torch.stack(data1[feature]).reshape(N, -1), torch.stack(data2[feature]).reshape(L, -1), p=p)
        else: 
            M = torch.cdist(torch.stack(data1[feature]).reshape(N, -1), torch.stack(data2[feature]).reshape(L, -1), p=p)

    M_normalized = M / M.max()

    M_normalized = M_normalized.to(torch.float64)
    
    a = torch.ones(N, dtype=torch.float64, device=M_normalized.device) * (1 / N)
    b = torch.ones(L, dtype=torch.float64, device=M_normalized.device) * (1 / L)
    if reg == None:
        W = ot.emd(a, b, M_normalized)
    else:
        W = ot.sinkhorn(a, b, M_normalized, reg = reg, method = method, numItermax=num_itermax, verbose=is_verbose, warn=True)

    return W

def _process_image(image, normalization):
    # passing in the same matrix that we pass into build transform if we normalized! 
    normalize = transforms.Normalize(*normalization)
    mean = torch.tensor(normalize.mean).view(-1, 1, 1)
    std = torch.tensor(normalize.std).view(-1, 1, 1)
    image = image * std + mean

    return torch.clamp(image, 0, 1)


def evalOT(
    data1, data2,
    feature='xs',
    method='sinkhorn_epsilon_scaling',
    reg=0.1,
    num_itermax=100_000,
    is_verbose=False,
    show_img=False,
    argmax=False,
    normalization=None, 
    dist_type=None, 
    p = 2, is_sq = True
):
    
    W = computeOT(
        data1, data2,
        feature=feature, method=method, reg=reg,
        num_itermax=num_itermax, is_verbose=is_verbose, 
        dist_type=dist_type, p = p, is_sq = is_sq
    )

    device = W.device
    y1_orig = torch.as_tensor(data1['ys'], device=device, dtype=torch.long)
    y2_orig = torch.as_tensor(data2['ys'], device=device, dtype=torch.long)

    row_sums = W.sum(dim=1, keepdim=True).clamp_min(1e-40)
    W_norm = W / row_sums

    unique_labels = torch.unique(torch.cat([y1_orig, y2_orig]), sorted=True)
    y1 = torch.searchsorted(unique_labels, y1_orig)
    y2 = torch.searchsorted(unique_labels, y2_orig)
    K = unique_labels.numel()

    Y2_onehot = torch.nn.functional.one_hot(y2, num_classes=K).to(dtype=W.dtype) 
    class_probs = W_norm @ Y2_onehot                                             
    
    if argmax:
        top1_idx = W.argmax(dim=1)   
        y_pred = y2[top1_idx]
    else: 
        if K == 2:
            y_pred = (class_probs[:, 1] >= 0.5).to(torch.long)
        else:
            y_pred = class_probs.argmax(dim=1)

    hits = (y_pred == y1)
    accuracy = hits.double().mean().item()
    accurate = int(hits.sum().item())
    print("number of accurate matches:", accurate)
    print("fraction accurate:", accuracy)

    cm_idx = y1 * K + y_pred
    conf_mat = torch.bincount(cm_idx, minlength=K*K).reshape(K, K).int()

    if show_img:
        topk_idx = W.topk(5, dim=1).indices  

        print("ACCURATE MATCHES")
        for i in hits.nonzero(as_tuple=False).squeeze(1).tolist():
            if i % 10 == 0:
                fig, axs = plt.subplots(1, 6, figsize=(10, 3.5))
                if normalization is not None:
                    img1 = _process_image(data1["xs"][i], normalization)
                else:
                    img1 = data1["xs"][i]
                img1 = img1.cpu()
                if img1.shape[0] == 1:  
                    axs[0].imshow(img1.squeeze(0).numpy(), cmap="gray")
                else:  
                    axs[0].imshow(img1.permute(1, 2, 0).numpy())

                axs[0].set_title(f"Query Image\nLabel={data1['ys'][i]}")
                axs[0].axis("off")

                for j in range(5):
                    col = topk_idx[i, j].item()
                    if normalization is not None:
                        img = _process_image(data2["xs"][col], normalization)
                    else:
                        img = data2["xs"][col]

                    img = img.cpu()  
                    if img.shape[0] == 1:  
                        axs[j+1].imshow(img.squeeze(0).numpy(), cmap="gray")
                    else:  
                        axs[j+1].imshow(img.permute(1, 2, 0).numpy())    
                        
                    matched_label = int(y2[col].item())
                    prob_val = W_norm[i, col].item()
                    axs[j+1].set_title(
                        f"Match {j+1}\nLabel={data2['ys'][col]}\nP={prob_val:.8f}"
                    )
                    axs[j+1].axis("off")
                for ax in axs: ax.axis("off")
                plt.tight_layout(); plt.show()

        print("INACCURATE MATCHES")
        for i in (~hits).nonzero(as_tuple=False).squeeze(1).tolist():
            if i % 10 == 0:
                fig, axs = plt.subplots(1, 6, figsize=(10, 3.5))

                if normalization is not None:
                    img1 = _process_image(data1["xs"][i], normalization)
                else:
                    img1 = data1["xs"][i]
                img1 = img1.cpu()
                if img1.shape[0] == 1:
                    axs[0].imshow(img1.squeeze(0).numpy(), cmap="gray")
                else:
                    axs[0].imshow(img1.permute(1, 2, 0).cpu().numpy())
                axs[0].set_title(f"Query Image\nLabel={data1['ys'][i]}")
                axs[0].axis("off")

                for j in range(5):
                    col = topk_idx[i, j].item()
                    if normalization is not None:
                        img = _process_image(data2["xs"][col], normalization)
                    else:
                        img = data2["xs"][col]
                    img = img.cpu()
                    if img.shape[0] == 1:  
                        axs[j+1].imshow(img.squeeze(0).numpy(), cmap="gray")
                    else:
                        axs[j+1].imshow(img.permute(1, 2, 0).cpu().numpy())
                    matched_label = int(y2[col].item())
                    prob_val = W_norm[i, col].item()
                    axs[j+1].set_title(
                        f"Match {j+1}\nLabel={data2['ys'][col]}\nP={prob_val:.8f}"
                    )
                    axs[j+1].axis("off")
                for ax in axs: ax.axis("off")
                plt.tight_layout(); plt.show()
    
    labels = [str(int(l.item())) for l in unique_labels]
    df_cm = pd.DataFrame(conf_mat.detach().cpu().numpy(), index=labels, columns=labels)
    plt.figure(figsize=(max(6, K), max(5, K)))
    sns.heatmap(df_cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    
    return accuracy, conf_mat

def test_ot_once(data1, data2, 
                 feature = 'xs', 
                 method = 'sinkhorn_epsilon_scaling', 
                 reg = 0.1, 
                 num_itermax = 100_000, 
                 is_verbose = False, 
                 return_confmat =True,
                 argmax = False, 
                 dist_type = None, 
                 p = 2, 
                 is_sq = True):
    N = len(data1['ys'])
    L = len(data2['ys'])

    if dist_type is not None:
        M = ssd.cdist(torch.stack(data1[feature]).reshape(N, -1), torch.stack(data2[feature]).reshape(L, -1), metric=dist_type)
        M = torch.tensor(M)
    else:
        if p == 2:
            if is_sq:
                M = torch.cdist(torch.stack(data1[feature]).reshape(N, -1), torch.stack(data2[feature]).reshape(L, -1), p=2) ** 2
            else: 
                M = torch.cdist(torch.stack(data1[feature]).reshape(N, -1), torch.stack(data2[feature]).reshape(L, -1), p=2)
        else: 
            M = torch.cdist(torch.stack(data1[feature]).reshape(N, -1), torch.stack(data2[feature]).reshape(L, -1), p=p)
                
    print("done with computing matrix")
    M_normalized = M / M.max()
    M_normalized = M_normalized.to(torch.float64)
    a = torch.ones(N, dtype=torch.float64, device=M_normalized.device) * (1 / N)
    b = torch.ones(L, dtype=torch.float64, device=M_normalized.device) * (1 / L)
    
    t0 = time.perf_counter()
    if reg == None:
        W, log = ot.emd(a, b, M_normalized, log=True)
        niter = None
        last_errs = None
    else:
        W, log = ot.sinkhorn(a, b, M_normalized, reg = reg, method = method, numItermax=num_itermax, verbose=is_verbose, log=True, warn = True)
        errs = log.get("err", [])
        last5 = errs[-5:]
        last_errs = [float(e) if not torch.is_tensor(e) else float(e.item()) for e in last5]
        niter = int(log.get("niter", 0)) if "niter" in log else None
    
    elapsed = time.perf_counter() - t0

    device = W.device
    y1_orig = torch.as_tensor(data1['ys'], device=device, dtype=torch.long)
    y2_orig = torch.as_tensor(data2['ys'], device=device, dtype=torch.long)

    row_sums = W.sum(dim=1, keepdim=True).clamp_min(1e-40)
    W_norm = W / row_sums

    unique_labels = torch.unique(torch.cat([y1_orig, y2_orig]), sorted=True)
    y1 = torch.searchsorted(unique_labels, y1_orig)
    y2 = torch.searchsorted(unique_labels, y2_orig)
    K = unique_labels.numel()

    Y2_onehot = torch.nn.functional.one_hot(y2, num_classes=K).to(dtype=W.dtype) 
    class_probs = W_norm @ Y2_onehot                                             
    
    if argmax:
        top1_idx = W.argmax(dim=1)   
        y_pred = y2[top1_idx]
    else: 
        if K == 2:
            y_pred = (class_probs[:, 1] >= 0.5).to(torch.long)
        else:
            y_pred = class_probs.argmax(dim=1)

    hits = (y_pred == y1)
    accuracy = hits.double().mean().item()
    accurate = int(hits.sum().item())
    print("number of accurate matches:", accurate)
    print("fraction accurate:", accuracy)

    cm_idx = y1 * K + y_pred
    conf_mat = torch.bincount(cm_idx, minlength=K*K).reshape(K, K).int()

    metrics = {
        "n": N,
        "correct": int(accurate),
        "accuracy": float(accuracy),
        "niter": None if niter is None else int(niter),
        "last_errs": None if last_errs is None else last_errs,
        "elapsed_sec": float(elapsed), 
        "method": method,
        "reg": None if reg is None else float(reg),
        "feature": feature,
        "p_norm" : int(p), 
        "is_sq" : is_sq,
        "norm_type" : dist_type if dist_type is not None else 'sqeuclidean',
    }
    
    if return_confmat:
        metrics["confmat"] = conf_mat.cpu().tolist()
        
    return metrics

## these are for plotting conf matrices 
def get_img_np(x):
    if torch.is_tensor(x):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr /= arr.max()

    return arr

def build_rep_images(dataset, labels: Optional[tuple] = None):
    reps = {}

    if labels is not None:
        label_to_idx = {int(l): i for i, l in enumerate(labels)}
    else:
        label_to_idx = None

    for img_tensor, label in dataset:
        lab = int(label)
        idx = label_to_idx[lab] if label_to_idx is not None else lab
        if idx not in reps:
            reps[idx] = img_tensor
            if label_to_idx is not None and len(reps) == len(labels):
                break
    return reps

def plot_conf_mat(conf_mat, rep_images1, rep_images2, name, scale=0.3, extra_band=1.2):

    if isinstance(conf_mat, str):
        conf_mat = ast.literal_eval(conf_mat)
    conf_mat = np.asarray(conf_mat)
    n = conf_mat.shape[0]

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n)))

    sns.heatmap(conf_mat, annot=True, fmt=".0f", cbar=False,
                annot_kws={"size": 15}, ax=ax)

    ax.set_title(f"Confusion Matrix — {name}", fontsize=20, pad=6)
    ax.set_xlabel("Predicted", fontsize=15)
    ax.set_ylabel("True", fontsize=15)
    ax.tick_params(axis='both', labelsize=15)

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)

    ax.set_xlim(0, n)
    ax.set_ylim(n, -extra_band)

    for i in range(n):
        if i in rep_images1:
            img = get_img_np(rep_images1[i])
            im = OffsetImage(img, cmap='gray' if img.ndim == 2 else None, zoom=scale)
            ab = AnnotationBbox(im, (n, i + 0.5),
                                xybox=(35, 0),  # push right
                                xycoords='data', boxcoords="offset points",
                                frameon=False, pad=0.0, clip_on=False)
            ax.add_artist(ab)

    y_top = -extra_band / 2.0
    for j in range(n):
        if j in rep_images2:
            img = get_img_np(rep_images2[j])
            im = OffsetImage(img, cmap='gray' if img.ndim == 2 else None, zoom=scale)
            ab = AnnotationBbox(im, (j + 0.5, y_top),
                                xycoords='data',
                                frameon=False, pad=0.0, clip_on=False)
            ax.add_artist(ab)

    plt.tight_layout()
    plt.show()
    return fig, ax

def plot_conf_mat_prob(conf_mat, rep_images1, rep_images2, title, scale=0.3, extra_band=1.2):
    if isinstance(conf_mat, str):
        conf_mat = ast.literal_eval(conf_mat)
    conf_mat = np.asarray(conf_mat)

    row_sums = conf_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  
    conf_prob = conf_mat / row_sums

    n = conf_prob.shape[0]

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n)))

    sns.heatmap(conf_prob, annot=True, fmt=".4f", cbar=False,
                annot_kws={"size": 10}, ax=ax)

    ax.set_title(f"{title}", fontsize=20, pad=6)
    ax.set_xlabel("Predicted", fontsize=15)
    ax.set_ylabel("True", fontsize=15)
    ax.tick_params(axis='both', labelsize=15)

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)

    ax.set_xlim(0, n)
    ax.set_ylim(n, -extra_band)

    for i in range(n):
        if i in rep_images1:
            img = get_img_np(rep_images1[i])
            im = OffsetImage(img, cmap='gray' if img.ndim == 2 else None, zoom=scale)
            ab = AnnotationBbox(im, (n, i + 0.5),
                                xybox=(35, 0),  # push right
                                xycoords='data', boxcoords="offset points",
                                frameon=False, pad=0.0, clip_on=False)
            ax.add_artist(ab)

    y_top = -extra_band / 2.0
    for j in range(n):
        if j in rep_images2:
            img = get_img_np(rep_images2[j])
            im = OffsetImage(img, cmap='gray' if img.ndim == 2 else None, zoom=scale)
            ab = AnnotationBbox(im, (j + 0.5, y_top),
                                xycoords='data',
                                frameon=False, pad=0.0, clip_on=False)
            ax.add_artist(ab)

    plt.tight_layout()
    plt.show()
    return fig, ax
    