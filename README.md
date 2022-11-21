# Deep-Learning-Systems
[CMU10714 - Deep Learning Systems: Algorithms and Implementation](https://dlsyscourse.org/)

**Implementation courses**

- Lec5: automatic differentiation
- Lec8: neural network library
- Lec13: hardware acceleration
- 

**Homework Summary**

- Hw0: prior knowledge review
  <span style="color:blue">*cross-entropy loss, SGD, softmax regression, two-layer nn* </span>
  
- Hw1: build a basic **automatic differentiation** framework

  <span style="color:blue"> backprogation, topological sort, reverse mode differentiation, SGD for two-layer nn</span>

- Hw2: implement a **neural network library** in the needle framework

  - Weight initialization: Xavier and Kaiming
  - Modules: Linear, ReLu, Sequentail, LogSumExp, SoftmaxLoss, Normalizaiton(Layer/Batch), Flatten, Dropout, Residual
  - Optimizers: SGD, Adam
  - Data primitives: `Dataloader` and `Dataset`
  - Build and train MLP ResNet

  