# Convolutional Networks

*Date: 2022/11/21*

---

**Problem with MLP**

- Number of parameters

  $\mathbf{x} \in \mathbb{R}^{WHC}$ leads to the weight matrix of size $(W\times H\times C)\times D$ where $D$ is the number of hidden units.

  256x256 RGB image ⟹ ~200K dimensional input ⟹ mapping to 1000 dimensional hidden vector requires 200M parameters (for only single layer).

  ![image-20221121125021838](C:\Users\Steve\AppData\Roaming\Typora\typora-user-images\image-20221121125021838.png)

- Fail to capture **translation invariance**.

  shifting image one pixel leads to very different next layer

## Convolution

 **Advantages**

- Share weights across all spatial locations $\rightarrow$ Drastically reduces the parameter count

  256x256 grayscale image ⟹ 256x256 single-channel hidden layer

  Requires only 9 parameters in 3x3 convolution.

- Convolution is a linear operator $\rightarrow$ translation invariance

  ![image-20221121125722193](C:\Users\Steve\AppData\Roaming\Typora\typora-user-images\image-20221121125722193.png)

**Convolution in 2d**

Progress as: "flip" the weight matrix $w$(filter with kernel size $k$), "slide" over the image to produce a **feature map**.
$$
\mathbf{Y} = \mathbf{W}  \ast \mathbf{X}
$$
  ![image-20221121143411896](C:\Users\Steve\AppData\Roaming\Typora\typora-user-images\image-20221121143411896.png)

**Convolution in deep networks**

- *Multi-channel convolutions*: map multi-channel inputs to multi-channel hidden units
  ![image-20221121150631761](C:\Users\Steve\AppData\Roaming\Typora\typora-user-images\image-20221121150631761.png)

  $$
    z[:,:, s]=\sum_{i n}^{c_{i n}} x[:,:, r] * W[r, s,:,:]
  $$

    \- $x \in \mathbb{R}^{h \times w \times c_{i n}}$ denotes $c_{i n}$ channel, size $h \times w$ image input

    \- $z \in \mathbb{R}^{h \times w \times c_{\text {out }}}$ denotes $c_{\text {out }}$ channel, size $h \times w$ image input

    \- $W \in \mathbb{R}^{c_{i n} \times c_{\text {out }} \times k \times k}$ (order 4 tensor) denotes convolutional filter

- Real implementation in matrix-vector form

   ![image-20221121151633495](C:\Users\Steve\AppData\Roaming\Typora\typora-user-images\image-20221121151633495.png)

**Padding**

To produce the output with the same size as the input.

![image-20221121151852617](C:\Users\Steve\AppData\Roaming\Typora\typora-user-images\image-20221121151852617.png)

Zero padding: pad input with $(k − 1)/2$ zeros on all sides.

**Pooling**

Incorporate max or average pooling layers to aggregate information

**Strided Convolutions**

Slide convolutional filter over image in increments >1 (= stride)

![image-20221121160456226](C:\Users\Steve\AppData\Roaming\Typora\typora-user-images\image-20221121160456226.png)

**Dilated convolution**

Increased receptive fields without increasing the number of parameters or the amount of compute

![image-20221121161504379](C:\Users\Steve\AppData\Roaming\Typora\typora-user-images\image-20221121161504379.png)

**Differentiate convolution** 

Define $z = conv(x, W) \rightarrow$ partial adjoints $\bar{v} \frac{\partial \operatorname{conv}(x, W)}{\partial W}, \quad \bar{v} \frac{\partial \operatorname{conv}(x, W)}{\partial x}$

  - Convolution as matrix multiplication

    Consider matrix-vector product $z = Wx$, then $\frac{\partial{z}}{\partial{x}} = W$, computing the backwards pass requires multiplying by $W^{T}\overline{v}$
    $$
    \left[\begin{array}{l}
    
    z_1 \\
    
    z_2 \\
    
    z_3 \\
    
    z_4 \\
    
    z_5
    
    \end{array}\right]=x * w=\left[\begin{array}{ccccc}
    
    w_2 & w_3 & 0 & 0 & 0 \\
    
    w_1 & w_2 & w_3 & 0 & 0 \\
    
    0 & w_1 & w_2 & w_3 & 0 \\
    
    0 & 0 & w_1 & w_2 & w_3 \\
    
    0 & 0 & 0 & w_1 & w_2
    
    \end{array}\right]\left[\begin{array}{l}
    
    x_1 \\
    
    x_2 \\
    
    x_3 \\
    
    x_4 \\
    
    x_5
    
    \end{array}\right] \quad \widehat{W}^T=\left[\begin{array}{ccccc}
    
    w_2 & w_1 & 0 & 0 & 0 \\
    
    w_3 & w_2 & w_1 & 0 & 0 \\
    
    0 & w_3 & w_2 & w_1 & 0 \\
    
    0 & 0 & w_3 & w_2 & w_1 \\
    
    0 & 0 & 0 & w_3 & w_2
    
    \end{array}\right]
    $$
    $\hat{W}^{T}\overline{v}$ is convolution with the flipped filter $\left[\begin{array}{lll}w_3 & w_2 & w_1\end{array}\right]$

    **Takeaway1**: $\bar{v} \frac{\partial \operatorname{conv}(x, W)}{\partial x} \Rightarrow$ convolve $\overline{v}$ with the flipped $W$!
    $$
    \left[\begin{array}{l}
    
    z_1 \\
    
    z_2 \\
    
    z_3 \\
    
    z_4 \\
    
    z_5
    
    \end{array}\right]=x * w=\left[\begin{array}{ccc}
    
    0 & x_1 & x_2 \\
    
    x_1 & x_2 & x_3 \\
    
    x_2 & x_3 & x_4 \\
    
    x_3 & x_4 & x_5 \\
    
    x_4 & x_5 & 0
    
    \end{array}\right]\left[\begin{array}{l}
    
    w_1 \\
    
    w_2 \\
    
    w_3
    
    \end{array}\right]
    $$
    **Takeaway2**: $\quad \bar{v} \frac{\partial \operatorname{conv}(x, W)}{\partial W} \Rightarrow$ multiplying by the transpose of this x-based matrix. 

