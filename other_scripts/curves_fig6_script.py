import numpy as np
import qupid.utils as spu
from qupid.qupid import QuPID
import warnings
warnings.filterwarnings('ignore')

# Dataset
dataset = "ORBIT5K"

# Data Generation
spu.compute_persistence(dataset)

# Data Processing
diagrams_dict, labels, n_data = spu.get_data(dataset)

# Diagrams Embedding
samplesH0, samplesH1 = spu.process_diagrams(diagrams_dict)
max_point0, max_point1 = spu.max_measures({"H0": samplesH0}), spu.max_measures({"H1": samplesH1})

# Model configuration and training
parameter_configs = [
    {"function": "id", "wave": " "},
    {"function": "fft", "wave": " "},
    {"function": "wvt", "wave": "coif2"},
    {"function": "wvt", "wave": "db3"}
]

alpha_opt = {"id": {"h0": (0, 0), "h1": (0, 0)},
             "coif2": {"h0": (0, 5e3), "h1": (6e2, 2e3)},
             "fft": {"h0": (0, 1e3), "h1": (0, 1e2)},
             "db3": {"h0": (0, 1e2), "h1": (5e3, 1e2)}}

RES = np.arange(1, 65)

for params in parameter_configs:
    transform = params["function"] if params["function"] != "wvt" else params["wave"]
    results = []
    for resolution in RES:
        modelH0 = QuPID(**params, resolution=(1, resolution), global_max=max_point0, global_min=(0, 0), alpha=alpha_opt[transform]["h0"])
        modelH1 = QuPID(**params, resolution=(resolution, resolution), global_max=max_point1, global_min=(0, 0), alpha=alpha_opt[transform]["h1"])
        data = []
        
        for i in range(n_data):
            modelH0.fit([samplesH0[i]])
            modelH1.fit([samplesH1[i]])
            data.append(np.concatenate((modelH0.transform(samplesH0[i]), modelH1.transform(samplesH1[i]))))

        mean, std = spu.evaluate_classifier_orbits(data, labels, verbose=False)
        results.append((resolution, mean, std))

    with open(f"results/{transform}_results.txt", "w") as f:
        for resolution, mean, std in results:
            f.write(f"Resolution: {resolution}, Mean: {mean}, Std: {std}\n")
