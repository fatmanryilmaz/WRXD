# WRXD
Weighted Reed Xiaoli Detector implementation for anomaly detection on hyperspectral images.

* Concentric dual windows applied to get local neighborhoods.
* Weights are obtained from GRX Detector.

AUC: 0.867 obtained from Salinas dataset (window sizes: 5,7).

Implementation based on the paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6782328

Some other HSI datasets for anomaly detection:

ABU (Airport-Beach-Urban) datasets: http://xudongkang.weebly.com/data-sets.html
