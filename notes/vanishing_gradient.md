# Vanishing Gradient Problem

*DATE: 2022/10/14*         Rephrased from [towards data science](https://towardsdatascience.com/the-vanishing-gradient-problem-69bf08b15484)

**Problem statement**

As more layers using certain activation functions are added to neural networks, the gradients of the loss function approaches zero, making the network hard to train.

**Cause**

Certain activation functions, like the sigmoid function, squishes a large input space into a small output space between 0 and 1 (**excessive saturation of activation functions**). Therefore, a large change in the input of the sigmoid function will cause a small change in the output. Hence, the derivative becomes small.

![img](https://miro.medium.com/max/770/1*6A3A_rt4YmumHusvTvVTxw.png)

Gradients of neural networks are found using backpropagation. By chain rule, the small derivatives of the last layer would affect its previous layers (n small derivatives multiplied together for n hidden layers). A small gradient means that the parameters of the initial layer will not be updated effectively during each epoch.

**Solutions**

- Residual networks
- Batch normalization

![img](https://miro.medium.com/max/770/1*XCtAytGsbhRQnu-x7Ynr0Q.png)