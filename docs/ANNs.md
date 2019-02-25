---
layout: default
title: Artificial Neural Networks
nav_order: 2
has_children: true
permalink: /docs/anns
---
# Chapter 3
# Artificial Neural Networks
**Use cases ANNs work well with:**
1. Instances are represented by many attribute-value pairs.
2. The target function output may be discrete-valued, real-valued, or a vector of several real- or discrete-valued attributes.
3. The training examples may contain errors.
4. Long training times are acceptable.
5. Fast evaluation of the learned target function may be required. Although the validation is quick.
6. The ability of humans to understand the learned target function is not important.


## PERCEPTRONS
Perceptrons are one of the simple *learning* models.
Given the input $$x_1$$ through $$x_n$$, the output $$h_{\theta}\left({}x_1 ...x_n\right)$$ is defined as:

$$h_{\theta}\left({}x_1 ...x_n\right) =  \begin{cases}
      1 & \text{if $w_0 + w_1 x_1 + ... + w_n x_n > 0$}\\
      -1 & \text{otherwise}\\
                                        \end{cases}$$
                                        
Or better yet, we can use the vector notation, and the hypothesis funciton can be succintly expressed as:

$$h_{\theta}\left({}x_1 ...x_n\right) = sgn\left(\overrightarrow{\boldsymbol{w}} . \overrightarrow{\boldsymbol{x}}\right)$$

where, $$sgn\left(y\right) = \begin{cases}
    1 & \text{if y > 0}\\
    -1 & \text{otherwise}
                            \end{cases}$$

**Comments:** Learning a perceptron is essentially figuring out the correct set of weights $$w_0, ..., w_n$$. For the sake of completeness, the space $$H$$ of candidate hypotheses in perceptron learning is the set of  all possible real-valued $$N$$ dimensional vectors.

**REPRESENTATIONAL POWER OF PERCEPTRONS**
One way to look at a perceptron is its geometrical interpretation- a hyperplance separating the two classes. The equation of the hyperplane itself is given by $$\boldsymbol{\overrightarrow{w}}. \boldsymbol{\overrightarrow{x}} = 0$$.

From a computational perspective, a single perceptron can learn to represent all the primitive boolean functions AND, OR, NAND and NOR. But every boolean function can be represented using multiple perceptrons. ```<formalize and insert proof here>```

**THE PERCEPTRON TRAINING RULE**
Training (or alternatively known as the learning rule for the perceptron) is precisely the problem of learning/ determining the wight vector that cause the perceptron to give the correct output (+- 1).

There are two commonly used algorithms:
1. **The perceptron rule:**
    We intialize the weights with random values, and modify them whenever the perceptron misclassifies a training example. We iterate the checking and updating rule over the entire training set. The weights are modified using the ***perceptron training rule*** described below:
    >$$w_i \gets w_i + \Delta w_i$$
where,
$$\Delta w_i = \eta (t-o)x_i$$
where, $$t$$ is the target output and $$o$$ is the predicted output, and $$\eta$$ is the learning rate.
> ```Insert the theorem of guaranteed covergence \cite{Minsky and Papert 1969}```
2. **The delta rule:**
The delta rule is designed to avoid the problem of finding weights when the training data is not linearly separable. Underlying the algorithm, is the concept of **gradient descent**. Additionally, gradient descent provides the basis for the famous **backpropagation** algorithm in training various neural network architectures. As will be clear later, the gradient descent has to be applied to the description of $$h_w(\overrightarrow{x})$$ as:
$$h_w(\overrightarrow{x}) = \overrightarrow{w}. \overrightarrow{x}$$

    Inorder to derive a weight learning rule, we specify an error metric, i.e. a measure for the *training error* of a hypothesis. This is not the only way, but nonetheless convenient.
    $$E(\overrightarrow{w}) = \frac{1}{2} \sum_{d\in D} \left(t_d - o_d\right)^{2}$$, where $$D$$ is the training dataset.
    
    If $$E$$ is the measure of error for the given $$w$$, then the partial derivative of $$E$$, i.e. $$\triangledown E$$, with respect to each weight gives the slope towards increasing error. This partial derivative of $$E$$ with respect to the vector $$w$$ is called **gradient** of E. Since the gradient specifies the direction of the steepest increase of $$E$$, the training rule becomes:
    >$$w_i \gets w_i + \Delta w_i$$
    where,
    $$\Delta \overrightarrow{w} = -\eta \triangledown E (\overrightarrow{w})$$
    For algorithm descign purposes, the training rule can be written in terms of individual weights too.
    
**HACKING THE TRAINING RULE**
The key practical difficulties in applying gradient descent are:
1. Converging to a local minima is sometimes quite slow.
2. In case of multiple local minima, it is not guranteed that the perceptron will find the global minimum.

One simple helpful technique is to use what is **incremental gradient descent** or alternatively, **stochastic gradient descent**. The idea is to update the weights with every training example. We change the learning algorithm accordingly.

# MULTILAYER NETWORKS AND THE BACKPROPATION ALGORITHM
We've seen two bases for learning- a linear function and the perceptron unit. Both pose problems to the learning of the multilayer network. 
1. No matter how many linear funcitons we used one after another, we end with a linear system. Simply speaking, they can't capture a non-linear approximation.
2. The perceptron rule is not differentiable therefore we can't use gradient descent to learn the weights.

**Sigmoid** functions, besides various other advantages, solve both the problems. They are non-linear as well as differentiable. We update our single neural units as:
$$h_w = \sigma\left(\overrightarrow{w}. \overrightarrow{x}\right)$$
where,
$$\sigma(y) = \frac{1}{1 + e^{-y}}$$

**THE BACKPROPAGATION ALGORITHM**
In a multilayer network with fixed set of units and interconnection, the backpropagation algorithm employs the gradient descent algorithm to attempt to minimize the mean square error between the actual values and the predictions for a given test set.
For a given set of weights, we define $$E$$ as sum of the errors over all of the network output units.

$$E(\overrightarrow{w}) = \frac{1}{2} \sum_{d\in D}{\sum_{k \in outputs}{ \left(t_{kd} - o_{kd}\right)^{2}}}$$


For each training example,

1.Propagate the input forward through the network, and calculate the $$o$$ for every node in the network.
2. Propagate the errors back through the network:

> 2.1 For each network output unit *k*, calculate its error term $$\delta_k$$
$$\delta_k \leftarrow o_k (1-o_k)(t_k - o_k)$$
2.2 For each hidden unit $$h$$, calculate its error term $$\delta_h$$
$$\delta_h \leftarrow o_h (1-o_h) \sum_{k \in outputs}w_{kh} \delta_k$$
2.3 Update each node weights $$w_{ji}$$
$$w_{ji} \leftarrow w_{ji} + \Delta w_{ji}$$
where, $$\Delta w_{ji} = \eta \delta_j x_{ji}$$

**HACK 1 - ADDING MOMENTUM**
Helpful in avoiding local minima. Add a factor from the previous error.
$$\Delta w_{ji}(n) = \eta \delta_j x_{ji} + \alpha \Delta w_{ji}(n-1)$$
where $$\alpha$$ is called the **learning momentum** and follows the euqlity: $$0 \leq \alpha \lt 1$$

**HACK 2 - LEARNING IN ARBITRARY ACYCLIC NETWORKS**
The above mulitlayer network backpropagation algorithm can be easily extended to arbitrary depth networks. We only need to generalize the formula for calculating the $$\delta$$ rule.
Let $$m$$ be the layer number and $$\delta_r$$ be the value for a unit $$r$$ in layer m.
$$\delta_r = o_r (1-0_r) \sum_{s\in{layer\space m+1}} w_{sr}\delta{s}$$

**INTERESTING FACTS ABOUT FEEDFORWARD NETWORKS**
1. Every **boolean function** can be represented exactly two layers of input.
2. Every bounded continuous function can be approxiamted with arbitrarily small error with two layers of inputs.
3. Any **arbitrary function** can be approximated to an arbitrary accuracy by a network of three layers.
 

# APPENDIX: DERIVATION OF THE BACKPROPAGATION RULE
The goal is that for every training example $$d$$, every weight $$w_{ij}$$ is updated by adding to it $$\Delta w_{ij}$$

$$\Delta w_{ij} = -\eta \frac{\partial E_d}{\partial w_{ji}}$$

where, $$E_d (\overrightarrow{w}) = \frac{1}{2} \sum_{k \in outputs} (t_k - o_k)^{2}$$

To implement gradient descent, we are only left with calculating $$\frac{\partial E_d}{\partial w_{ji}}$$
By the chain rule, we get

$$\frac{\partial E_d}{\partial w_{ji}} = \frac{\partial E_d}{\partial net{j}} \frac{\partial net_j}{\partial w_{ji}}$$

$$\frac{\partial E_d}{\partial w_{ji}} = \frac{\partial E_d}{\partial net{j}}x_{ji}$$

> **CASE I: $$j$$ is an outer node.**
$$\frac{\partial E_d}{\partial net_{j}} = \frac{\partial E_d}{\partial o_{j}} \frac{\partial o_j}{\partial net_{j}}$$
where, $$\frac{\partial E_d}{\partial o_{j}} = -(t_j - o_j)$$ i.e. the differentiation of error metric.
and, $$\frac{\partial Eo_j}{\partial net_{j}} = -o_j(1 - o_j)$$,
finally, putting it all together,
we get, $$\Delta w_{ji} = \eta \frac{\partial E_d}{\partial w_{ji}} = \eta (t_j - o_j) o_j (1 - o_j) x _{ji}$$
> **CASE II: $$j$$ is a hidden layer node.**
$$\frac{\partial E_d}{\partial net_{j}} = \sum_{k \in Downstream(j)} \frac{\partial E_d}{\partial net_{k}} \frac{\partial net_k}{\partial net_{j}}$$
replacing, $$\frac{\partial E_d}{\partial net_{k}} =  \delta_k$$
we get, 
$$\frac{\partial E_d}{\partial net_{j}} = \sum_{k \in Downstream(j)}  \delta_k \space  \frac{\partial net_k}{\partial net_{j}}$$
$$\frac{\partial E_d}{\partial net_{j}} = \sum_{k \in Downstream(j)} \delta_k \space  \frac{\partial net_k}{\partial o_{j}} \frac{\partial o_j}{\partial net_j}$$
$$\frac{\partial E_d}{\partial net_{j}} = o_j (1-o_j)\sum_{k \in Downstream(j)} \delta_k \space  \frac{\partial net_k}{\partial o_{j}}$$
$$\frac{\partial E_d}{\partial net_{j}} = o_j (1-o_j)\sum_{k \in Downstream(j)} \delta_k \space  w_{kj}$$


