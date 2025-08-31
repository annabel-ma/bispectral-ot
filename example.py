# Evaluating Bispectral OT on MNIST where one half is rotated and the other half is not. Seed 0, ground metric for OT is L2. 

from data_processing import *
from ot_functions import *  
import torchvision 

DATA_DIR = Path.cwd() / "datasets"
DATA_DIR.mkdir(parents=True, exist_ok=True)
data = torchvision.datasets.MNIST(DATA_DIR, download = True, train=True, transform = build_transform(normalization=MNIST_NORM))

# Toggle rotate to False to have both sets unrotated, n returns the number of each class in each split
set1, set2, n = extract_disjoint_sets(data, seed=0, contiguous=False, rotate=True)

set1_prepped = prepare_bispec(set1)
set2_prepped = prepare_bispec(set2)

metrics = test_ot_once(
        set1_prepped, set2_prepped,
        feature="bs", #toggle to xs to run standard OT
        method="greenkhorn", # ot solver
        reg=0.01,
        is_verbose=True,
        return_confmat=True,
        argmax=False, # toggle to True to assign based on the argmax of each row of Gamma, rather than the label with the maximum total mass
        num_itermax=10_000_000, 
        p = 2, # if a euclidean p-norm 
        is_sq = False # can be toggled on to get squared euclidean distance, can also pass dist_type = "cosine" for example to get any of the scipy cdists
    )

print(metrics) # dictionary with accuracy, confusion matrix, etc.