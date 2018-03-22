# ALAD
Accelerated Local Anomaly Detection in Attributed Networks

The whole pipeline is separated into two parts:

(1) Run DSGD_CNMF.py, which implements CNMF and get factor matrices W and H

(2) Run DSGD_CNMF_FindOutlier.m to detect anomalies (An example result is in the ‘Disney’ folder, in which you can evaluate directly by calling the function)
