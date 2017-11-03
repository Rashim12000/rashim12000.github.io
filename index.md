## Predicting Drug Target Interaction Using Deep Belief Network
With the advancement in AI field, machine learning methods are being used to train the classifier for separating intractable drug-target pair as it is difficult to classify dockable and non-dockable ligands due to non-linear nature of big-biological data. As deep learning has been shown to produce state-of-the-art results on various tasks, we propose a new approach to predict the interaction between drug and targets efficiently. The DBN is used to extract the high level features from 2D chemical substructure represented in fingerprint format. DBN is trained in a greedy layer-wise unsupervised fashion and the result from this pre-training phase is used to initialize the parameters prior to BP used for fine tuning. Similarly, logistic regression layer is staked as output layer. Then it is fine-tuned using BP of error derivative to build classification model that directly predict whether a drug interacts with a target of interest or not. In addition to this we too propose an approach to reduce the time complexity of training the learning method with the use of GPU which is highly parallel programmable processor featuring peak arithmetic and memory bandwidth that substantially outpaces its CPU counterpart.


### Restricted Boltzmann Machine

An RBM is an energy-based probabilistic model, in which the Gibbs probability distribution is defined through an energy function. 
The graph of an RBM has connections only between the layer of hidden and the layer of visible variables, but not between two variables of the same layer. In terms of probability, this means that the hidden variables are independent given the state of the visible variables and vice versa.


### Matrix Multiplication in GPU
The GPU used in this project is based on CUDA, a GPU programming model from   NVIDIA. The CUDA programming model uses the CPU as the host and the GPU as the co­ processor or device. One host and several devices are present in a heterogeneous system. In such system, the CPU and the GPU work together to finish the task. The CPU is responsible for logical and sequential computing tasks, and the GPU focuses on the parallel task at the thread level. Once the parallel part is determined, we can assign the parallel task for the GPU. The parallel program on the GPU is called a kernel, which is not a complete program but a part of a program to finish the parallel task. In a heterogeneous system, the sequential program on the CPU and the parallel program on the GPU forms a complete program. To date, every kernel is executed by a grid with many blocks. A block has many threads and memory that is shared by the threads in the same block.

Matrix multiplication is an essential building block for numerous numerical algorithms, for this reason most numerical libraries implements matrix multiplication. One of the oldest and most used matrix multiplication implementation GEMM is found in the BLAS library. While the reference BLAS implementation is not particularly fast. There are a number of third party optimized BLAS implementations like CUBLAS from NVIDIA.
```markdown
 Public void cublasSgemm (char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
```
