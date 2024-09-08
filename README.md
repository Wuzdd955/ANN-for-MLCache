# ANN-for-MLCache

This repository contains a replication of the ANN-based MLCache mechanism as described in the paper:

Liu W, Cui J, Li T, et al. “A space-efficient fair cache scheme based on machine learning for NVMe SSDs.” IEEE Transactions on Parallel and Distributed Systems, 2022, 34(1): 383-399.

##Overview

This project replicates the machine learning-based cache scheme (MLCache) for NVMe SSDs. The scheme aims to improve cache efficiency and fairness by leveraging artificial neural networks (ANNs) to predict the manage cache behavior.

##Files Included

2mix-stream-training-data.csv: Training dataset containing data for second mixed streams.

4mix-stream-training-data.csv: Training dataset containing data for four mixed streams.

8mix-stream-training-data.csv: Training dataset containing data for eight mixed streams.

MLPRegression.py: The replication code.

