# Neural Network in Practice

*Date: 2022/11/05*

---

## Neural Network Library Abstractions

Programming abstractions of a framework

- defines the common ways to **implement, extend and execute** model computations
- case studies: Caffe1.0(2014) -> TensorFlow 1.0(2015) -> PyTorch(2016)

![image-20221105214303405](C:\Users\Steve\AppData\Roaming\Typora\typora-user-images\image-20221105214303405.png)

-  `forward`: both top and bottom have the memory **pre-allocated**. Run the computation and copy data to top
- `backward`: `top` contains the gradient from the output. `propagte_down`: indicator whether to propagate a gradient to the input.
- backpropagation in place by reverse topological iterations

![image-20221105225823281](C:\Users\Steve\AppData\Roaming\Typora\typora-user-images\image-20221105225823281.png)

- First framework to declare the computational graph (concepts introduced by Theano).
- Declarative programming: `v1` is an in-place placeholder(we dont know its value when declare this variable)
- We construct the computation graph before any computation
- Create a `session` which helps send the execution command
- **Advantage**: gives the opportunity to look at only a part of computational graph and skip the unnecessary. Effectively having a complete computational graph gives you a lot of opportunities for optimization. Scalable computation having the execution distributed on different machines.

![image-20221106005016273](C:\Users\Steve\AppData\Roaming\Typora\typora-user-images\image-20221106005016273.png)

- Difference: "define-by-run" vs "run-after-define"
  - Optimization opportunities
  - allows more organic mixing of Python
  - Debug
- Deep thoughts: pros and cons of each abstractions. What kind of interface that you would design to resolve the problem of both modular construction of models, as well as expressing different kind of optimization algorithms and needs

## High level modular library components

- `nn.Module`: compose things together, tensor in -> tensor out.
  - get the list of parameters
  - ways to initialize the parameters
- Loss function
  - training and inference modes
- Optimizer
  - takes in a list of weights
  - keep tracks of auxiliary states
  - consider **regularization**