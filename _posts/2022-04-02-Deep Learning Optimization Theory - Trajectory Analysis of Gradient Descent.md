---
layout: post
comments: true
title:  "Deep Learning Optimization Theory - Trajectory Analysis of Gradient Descent"
date:   2022-04-02 10:10:10 +0300
tags: Deep-learning-theory Optimization
---
> A prominent approach in the study of deep learning theory in recent years has been analyzing the trajectories followed by gradient descent. This post is an introduction to this approach & my path to understanding it a little better.

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

# Introduction







As the exciting applications and capabilities of deep learning flourished in both academia and industry in recent years, the theoretical understanding of its success lags behind. More often than not, practitioners design deep learning-based solutions that are based on conventional wisdom and intuition, with no strong theoretical justification behind them.







In particular, in recent years an obvious yet mysterious fact that stood across various experiments is the ability of gradient descent, a relatively simple first-order optimization method, to optimize an enormous number of parameters on highly non-convex loss functions. In some sense, this practical observation stands in contrast to classical statistical learning theory. This post will discuss the significant progress researchers are making in bridging this theory gap and demystifying gradient descent.







Before diving into the details, let us distinguish between the two research approaches that gained the most interest in the past few years. These approaches are the *landscape approach* and the *trajectory approach*.







The **landscape approach** aims to analyze the loss function landscape (what extreme points does it have, and where are they? how smooth the function is?), and is motivated by empirically optimistic results [Goodfellow et al., 2014](https://arxiv.org/pdf/1412.6544.pdf) that conclude that "optimizing non-convex functions using gradient descent is somehow simpler than we thought". See [here](https://omrikaduri.github.io/2021/10/25/DL-Optimization-Introduction.html) an introduction to those experimental results and some theoretical investigation of them. Nevertheless, results from previous years ([Kawaguchi, 2016](https://arxiv.org/pdf/1605.07110.pdf)) indicate that for 3-layer neural networks, this approach fails to prove convergence. 







The **trajectory approach**, on the other hand, tries to understand the specific trajectories gradient descent follows during optimization. Therefore, even if the loss function itself might be highly non-convex, this approach hopes to find that (under mild assumptions) the trajectories taken are somewhat "nice" (vaguely defining "nice" as "possible to analyze"). Specifically, this approach wishes to analyze in which way will the trajectory follow and at what rate (will GD converge to a global minimum? after how many iterations?)







In this post, I will overview several interesting results from the trajectory approach from recent years. Specifically, I will highlight the results of **[Arora et al., 2018](https://arxiv.org/pdf/1810.02281.pdf)** on the convergence rate of gradient descent over deep linear neural networks. Then, I will wander about its assumptions and look to simplify them, by introducing the importance of the network's width, as done in **[Du et al., 2018](https://arxiv.org/pdf/1810.02054.pdf)**. I will observe that a crucial distinction between those two works lies in a specific choice - what is the trajectory we are analyzing? Specifically, we will observe that some works focus on the **parameters trajectory** (how the parameters change during optimization), and some works focus on the **predictions trajectory** (how predictions change during optimization). Finally, I will conclude by mentioning several other results from recent years that followed the trajectory approach and will highlight intriguing assumptions of this analysis.







<!-- The reader is assumed to be familiar with SGD, deep learning, and some background in optimization. -->

<!-- > Note that one can distinguish the approaches also by the tools they wish to utilize for the analysis of gradient descent. The first approach relies on geometry and function analysis, and the other on differential analysis.  -->







# High-Level Recipe for The Trajectory Approach



In recent years, research that follows the trajectory approach is flourishing and provides outstanding results that help both theoreticians and practitioners understand why and how gradient descent converges to a global minimum (For example, the Neural Tangent Kernel gives intuition to the positional encoding that is used in NeRF (more at [Tancik et al., 2020](https://arxiv.org/pdf/2006.10739.pdf))). However, it is not always clear what the exact relationships are between works that solve the same underlying problem.







Here I propose a high-level recipe that dissects the 3 different steps for reasoning about the optimization problem from the lens of the trajectory analysis approach. 







1. **Dynamics definition** - Define the dynamical system you wish to analyze. Start with the definition of the **state** (e.g., the parameters or the predictions at time **t**), and then define the **dynamics** that describes how the states evolve in time (e.g., how the parameters/predictions change). We wish to find the **"physics & constraints"** that gradient descent induces on the state we are analyzing (predictions or parameters, in our examples at this post).



2. **Convergence to a global minimum** - Prove that after T iterations from an initial state (e.g., randomly initialized NN) the gradient induced by the dynamics converges (i.e. loss < $$\epsilon$$ ). Typically referred to as proving that the loss is monotonically decreasing.



3. **Convergence at the desired rate** - Prove that T is linear/polynomial/etc. function of the desired performance (epsilon) and possibly the network's architecture (depth, width, etc.).







While this general recipe may not always hold, I find it useful to dissect what is essentially the goal behind the different works. Equipped with this mental recipe in mind, let us first observe the work from [Arora et al., 2018](https://arxiv.org/pdf/1810.02281.pdf).







# A Convergence Analysis of Gradient Descent for Deep Linear Neural Networks







One of the toughest questions that keeps optimization researchers busy in recent years is the specific effect that increasing the neural network's depth has on their success. Classical theory tells us that while increasing the model's complexity is beneficial in terms of expressivity, it complicates optimization. Therefore, there is an obvious tradeoff that needs to be adjusted carefully.







To dissect the specific effect of depth, this paper focused on **linear neural networks** (LNN). 
An LNN is a network that is defined by composing a number of (learned) linear transformation, i.e. $$y=W_N...W_1x$$. Since adding layers to an LNN does not modify the model's expressiveness (as the composition of multiple linear trasnformations can still only express a linear transformation), it is a handy testbed for analyzing the effect of *over-parameterization*. By *over-parameterization* we roughly mean having more parameters than is "needed" (which I will leave vaguely for now), and classical theory tells us that in this over-parameterization regime we are simply increasing the risk of over-fitting. 







While the expressivity does not change as depth increases, it is not the case in terms of optimization. Increasing depth of LNN leads to non-convex training problems with multiple minima and saddle points.

<details style="border: 0.5px black; border-style: dashed" markdown="1">
<summary><b>Intuition for the non-convexity of over-parameterized LNNs</b></summary>
The fact the over-parameterized LNNs lead to non-convex training is not trivial. The simple shallow layer linear neural network is known to be convex. Specifically, for the model $$y=Wx$$ and the quadratic loss, it is well-known that we can analytically find the solution, without even running optimization. But as greatly detailed in Section 3 of [Arora et al., 2018a](https://arxiv.org/pdf/1802.06509.pdf) , even over-parameterizing by a single scalar $$w_2$$, i.e. $$y=W_1W_2x$$ induces infinitely many global minimas.











Intuitively, imagine that you somehow found the parameters that minimize the loss, namely $$ W_1^*$$, $$W_2^*$$. Now you can just multiply the first by some constant *c* and multiply the other by its reciprocal, and there you have another solution. 







While this is not sufficient to show that this problem is non-convex (as the infinitely many global minima may be "close together" in a way that the loss does not change between them), it should provide you with the intuition that over-parameterization of LNNs induces a non-convex problem. 







</details>











This work proves that (under assumptions detailed below) gradient descent converges at a linear rate to a global minimum for deep linear neural networks. Let's dive into the work, which can be seen as tackling the three steps shown above.

## Dynamics of Gradient Descent for Deep LNN (Linear Neural Networks)

The paper describes the state of the system as the parameters and describes how those parameters evolve (i.e., the dynamics of the system). With this at hand, we can analyze the path the weights are taking "across time" and derive conclusions about the rate of convergence. However, dealing with discrete time steps (i.e., number of training iterations, which is discrete) is non-trivial and is much more involved.







Therefore, a recurring theme in the trajectory approach is "taking the learning rate ($$\eta$$) to zero", which induces a **continuous** learning algorithm - **gradient flow**. Gradient flow enables us to utilize tools from differential equations to analyze the optimization of neural networks. Specifically, recall that gradient descent updates the weights at each step by:







$$\begin{equation}



W_{j}(t+1) = W_{j}(t) - \eta \frac{\partial L^N}{\partial W_{j}}(W_1(t),...,W_N(t)) 



\tag{1}



\end{equation}



$$







where *$$W_j(t)$$* is the weight matrix corresponding to layer **$$j$$** (from 1 to $$N$$ layers) in time **t**. $$L^N$$ denotes the loss of an N-layer neural network. So far it is just a regular gradient descent update rule for a deep linear neural network with $$N$$ layers.







Let us rearrange the terms and recall that $$\eta \to 0$$:







$$\begin{equation}



\frac{W_{j}(t+1) - W_{j}(t)}{\eta} = -\frac{\partial L^N}{\partial W_{j}}(W_1(t),...,W_N(t))



\tag{2} 



\end{equation}$$







Note that the left side of the equation is essentially the **derivative of the weights with respect to time**. Also, let us define this derivative as $$\dot{W}_j(t)$$. If it feels awkward, just remember that when $$\eta \to 0$$ the left side term defines the infinitesimal change of $$W_j$$ in time (where we assume t and t+1 to represent two consecutive time steps), which is just a longer way of saying "derivative".







However, analyzing this equation is still not trivial. Specifically, the gradient of the loss with respect to each layer's weight matrix j is non-trivial, as the over-parameterized network loss (denoted as $$L^N$$) is non-convex. On the other side, we observe that the gradient of the loss function of the single hidden-layer network (denoted as $$L^1$$) is easier to analyze, as its loss is convex. Therefore, the authors seek to **tie the dynamics** of the single hidden-layer network to those of the over-parameterized network. In simpler terms, they are aiming to describe the dynamics of the non-convex over-parameterized network using the dynamics of the shallow network. 







First, they denote the equivalence of the losses $$L^N$$ and $$L^1$$. For a given matrix $$W_e$$ that can be decomposed as:







$$\begin{equation}



W_{e} = \Pi_{1}^{j=N}W_j



\tag{3}



\end{equation}



$$







The losses $$L^1$$ and $$L^N$$ are equal:



$$\begin{equation}



L^N(W_1,...,W_N) = L^1(W_e)



\tag{4}



\end{equation}



$$







Where $$W_e$$ are the weights of the shallow single hidden-layer linear neural network, and $$(W_1,...,W_N)$$ are the weights of the deep linear neural network.







Using matrix calculus (refer to this [cheet sheet](https://atmos.washington.edu/~dennis/MatrixCalculus.pdf) for a refresher) the gradient of $$L^N$$ with respect to the weights $$W_J$$ of layer $$j$$ can be described as:







$$\begin{equation}



\frac{\partial L^N}{\partial W_{j}}(W_1(t),...,W_N(t)) = \Pi_{i=j+1}^N W_i^T \frac{d L^1(W_e)}{dW_e}\Pi_{i=1}^{j-1}W_i^T



\tag{5}



\end{equation}



$$







<details style="
border: 0.5px black; border-style: dashed" markdown="1">
<summary><b> A 3-layer example of the derivation </b></summary>

For those of you that do not see the last derivative at (5) intuitive, let us quickly develop it for a simple 3-layer linear neural network example. In the context of this example, denote $$W_e = W_3W_2W_1$$. Let us see the derivative of the loss $$L^3$$ with respect to $$W_2$$:

$$\begin{equation}
\frac{\partial L^3}{\partial W_{2}}(W_1,W_2,W_3) = W_3^T \frac{d L^1(W_e)}{dW_e}W_1^T
\tag{5.1}
\end{equation}$$

Roughly speaking, since the network activations are linear, the derivation is simply taking all layers "deeper" than $$j$$ and multiplying by them from the left, and all layers "shallower" than $$j$$ and multiplying by them from the right. In the "middle", all we are left with is the derivative of the loss of the shallow network with respect to $$W_e$$, which is a simple convex loss.
</details>

However, we are still left with a non-trivial dynamical system to analyze, where the dynamics of every layer $$j$$ are tied to the state of all other layers. This is caused due to the appearance of the $$W_i$$ terms. 







Since the paper's proof is highly involved, I attempt to simplify things as much as I can while trying to not oversimplify. Let start from the end and understand what the desirable dynamics of the weights look like by investigating the results(*) of an earlier paper by [Arora et al., 2018a](https://arxiv.org/pdf/1802.06509.pdf), which they relied on. Using an assumption they called **balancendness**, and the assumption of gradient flow, they achieved to reveal the following dynamics:







$$\begin{equation}



% \dot{W_e} = \sum_{j=1}^N[]



vec(\dot{W_e(t)}) = -P_{W_e(t)}vec(\frac{dL^1(W_e)}{dW_e})



\tag{6}



\end{equation}



$$











> (\*) Note that I use a slightly different version of SGD than what was used in the original paper of [Arora et al., 2018a]((https://arxiv.org/pdf/1802.06509.pdf)). However, I believe that the main ideas persist, so I choose the "common" variant of gradient descent.







Well, what does that mean? The `vec(W)` is simply a vectorization of a matrix in column-first, so no magic happens there. But what is that $$P_{W_e(t)}$$ term? If we drop that term, we get the trivial dynamics for the case that N=1. However, when N>1, it is not the case, as we have seen above in the matrix form (before vectorization, in eq. 5). Therefore, one can say that the magic of over-parameterization (for the case of linear neural networks) hides in that term. Before trying to wrap our minds about it, simply observe that if eq. 6 holds, the dynamics of the weights $$W_e$$ are a simple ODE that governed by the gradient of the convex loss $$L^1$$ pre-multiplied by the matrix P. 







Quite surprisingly, this result stems from making a relatively reasonable assumption on the **initialization** (at t=0) of the parameters, called **balancedness**, for all layers:







$$\begin{equation}



W_{j+1}^T(0)W_{j+1}(0)=W_j(0)W_{j}^T(0)



\tag{7}



\end{equation}



$$







<!-- <details style="border: 0.5px black; border-style: dashed" markdown="1">>



<summary>Why is that assumption considered reasonable?</summary>







This assumption suggests that conventional initialization methods via random Gaussian distributions with mean zero leads to the (approximate) balancendness shown above. 



</details> -->







Actually, as stated by [Du et al., 2018a](https://proceedings.neurips.cc/paper/2018/file/fe131d7f5a6b38b23cc967316c13dae2-Paper.pdf), gradient descent keeps the difference between the norms of consequent layers (approximately) constant throughout the optimization. Therefore, if we assume practical Gaussian near-zero initialization, this assumption is easily made. To be slightly more formal, if eq. 7 holds at initialization, it keeps to hold throughout optimization:







$$\begin{equation}



W_{j+1}^T(t)W_{j+1}(t)=W_j(t)W_{j}^T(t)



\tag{8}



\end{equation}



$$











To understand the benefit of making this assumption, let us perform an SVD for both sides (and leave time notation for simplicity):







$$\begin{equation}



V_{j+1}\Sigma_{j+1}^TU_{j+1}^TU_{j+1}\Sigma_{j+1}V_{j+1}^T = U_j\Sigma_{j}V_{j}^{T}V_j\Sigma_{j}^TU_j^T



\end{equation}



$$







Simplifying the terms, we get:







$$\begin{equation}



V_{j+1}\Sigma_{j+1}^T\Sigma_{j+1}V_{j+1}^T = U_j\Sigma_{j}\Sigma_{j}^TU_j^T



\end{equation}



$$







<details style="border: 0.5px black; border-style: dashed" markdown="1">
<summary><b>A quick refresher on SVD for that derivation</b></summary>
If you are not sure why that transition happened, please watch [this great lecture](https://www.youtube.com/watch?v=TX_vooSnhm8&ab_channel=MITOpenCourseWare) on SVD by Prof. Gilbert Strang. 

In the meantime, for our purposes, it is sufficient to know that SVD is a method to decompose a matrix into 3 different matrices: $$U \Sigma V^T$$. Luckily, the matrices $$U$$ and $$V$$ in this decomposition are **orthonormal**, and that means that multiplying each of them by itself would result in the identity matrix $$I$$. Therefore, after performing an SVD we can simplify the equation and arrive to the last equation shown above.
</details>







That's interesting! Observe that the two sides are equal and therefore are clearly an eigendecomposition of the same matrix. And since **eigenvalues are unique**, we conclude that **all layers have the same set of singular values**. Moreover, each pair of eigenvectors from $$V_{j+1}$$ and $$U_j$$ are the same up to multiplication by a scalar. Therefore, the left singular vectors of each layer coincide (i.e. pointing in the same direction) with the right singular vectors of the layer that follows. In a way, it says that under the balancedness assumption, gradient descent pushes all weight matrices in the same direction. 







This result largely simplifies the product of all layers, when applied in a successive manner. This largely simplifies $$W_e$$, as the authors show in Appendix A.1 at [Arora et al., 2018](https://arxiv.org/pdf/1810.02281.pdf). Without diving further into the proof, I encourage you to go read the proof itself and keep in mind the two key takeaways: 



1. The dynamics are **simple and governed by the matrix P**, which is time-dependent and depth-dependent.


2. The balancendness assumption **simplified the dynamics**, without being too restrictive on the initialization technique.


> A slightly intriguing point that did not stand up the generalization to more practical scenarios from [Arora et al., 2018a](https://arxiv.org/pdf/1802.06509.pdf) is the following quote:
>"from an optimization perspective, overparameterizing using wide or narrow networks has the same effect – it is only the depth that matters."
> As we shall see, it stands somewhat in contrast to modern results that found the width to play a crucial effect on optimization.







## Convergence to Global Minimum ###







Following the high-level recipe presented at the beginning of this post, and equipped with a nice formulation of the dynamics that ties the dynamics of an over-parameterized linear neural network to a single hidden layer neural network, we wish to prove the convergence of gradient descent to a global minimum. 


However, now it is time to reveal another critical thing about the matrix $$P_{W_e(t)}$$ from eq. 6 - it is a **Positive Semi-Definite** (PSD) matrix. In our context, it means that $$v^TPv \ge 0$$ for every vector $$v$$ (with a slight abuse of notation, we assume $$v$$ is a column vector).   

To be a bit slightly more formal, note that it is sufficient to prove that the loss is a monotonically decreasing function. If so, it is obvious that in the limit, it would converge to zero. Furthermore, we are assuming that the loss is convex (as the function of $$y'$$) and non-negative, as with most modern neural networks loss functions (quadratic, logloss, etc.). Finally, note that as the losses $$L^1(W_e)$$ and $$L^N(W_1,..., W_N)$$ are equal in our case (where $$W_{e} = \Pi_{1}^{j=N}W_j$$), if we prove that the first is monotonically decreasing, it implies the same on the other.


Let us observe at the derivative of the loss with respect to time ($$\frac{dL(W_e)}{dt}$$). We wish it to be some negative number, as it would essentially mean that the slope is negative and the function is monotonically decreasing. Well, by the chain rule we know that: 

$$\begin{equation}

\frac{dL^1(W_e)}{dt} = \frac{dL^1(W_e)}{dW_e}\frac{dW_e}{dt}

\end{equation}$$







Since a vectorization is simply an arrangement of a matrix (and each of the terms is indeed a Jacobian(**) matrix) as a vector, it does not affect the result and therefore:

$$ \frac{dL^1(W_e)}{dt} = vec(\frac{dL^1(W_e)}{dW_e})^T vec(\frac{dW_e}{dt})$$

> (\**) If you are unfamiliar with the Jacobian, it might sound scary. But as you are probably familiar with the gradient operator (a vector of partial derivatives), simply note that the Jacobian generalizes the gradient to the case of multi-output function.

We saw that last term before, in eq. 6. Plugging it here we get:

$$ \frac{dL^1(W_e)}{dt} = -vec(\frac{dL^1(W_e)}{dW_e})^T P_{W_e(t)}vec(\frac{dL^1(W_e)}{dW_e})$$







By the fact that $$P_{W_e}(t)$$ is a PSD matrix, and recall that $$v^T P_{W_e}(t)v \ge 0$$, we conclude that:







$$ \frac{dL^1(W_e)}{dt} \le 0$$







That is (almost) sufficient to prove convergence to a global minimum in the limit (taking infinitely many gradient steps). This proof is obviously not complete (as we need to show that the function is strictly decreasing at least for some time for it to converge), but for our purpose of grasping the works of trajectory analysis, it is enough. For a more rigorous (and stronger) proof, read Lemma 1 in [Arora et al., 2018](https://arxiv.org/pdf/1810.02281.pdf).







## Convergence Rate of Gradient Descent for LNN ###



We already made 2 out of the 3 steps in the recipe! We defined our dynamical system for gradient descent and observed that the loss induced by it converges to a global minimum after an infinitely large amount of steps. It is time to tackle the third and final step and approve that the convergence occurs in a plausible time. Specifically, the authors prove linear **rate of convergence** for deep linear neural networks.







However, to deal with the convergence rate, we need to somehow **lower bound the loss decrease** in a way that would ensure a **non-trivial decrease** at every step. But how can we ensure that the loss decreases by at least some amount at each step?







Let us first note how a convergence rate is defined. Recall that the limit of the sequence of the loss values across time $$(L^1(t))_{t \in [0,\infty]}$$ is zero, i.e., $$\lim_{t \to \infty}L^1(t) =0$$. 



When the limit is zero, the [**rate** of convergence](https://en.wikipedia.org/wiki/Rate_of_convergence) to the limit is considered linear if the following holds:







$$\begin{equation}



\lim_{t \to \infty} \frac{|L(t+1)|}{|L(t)|} = \mu



\end{equation}$$







for $$\mu \in (0,1)$$ which is denoted as the rate of convergence. Specifically, since our losses are non-negative, the absolute term is not needed. To gain an intuition why this is called a "linear rate", simply observe that $$L(t+1)$$ can be approximated linearly by $$L(t)$$ by rearranging the terms - $$L(t+1) \approx L(t)\mu $$ . Also, unrolling this equation in time (starting from t to 0) reveals an exponential relation between the loss at time t and the initial loss: 







$$\begin{equation}



L^1(t) = L^1(0)\mu^t    



\end{equation}$$







Let us observe what is the $$\mu$$ the authors had found to satisfy that equation:







$$\begin{equation}



L^1(t) \le L^1(0)(1-\eta c^{\frac{2(N-1)}{N}})^t    



\end{equation}$$







This result is enough to prove linear convergence and therefore determine the number of iterations needed until converging to any desired loss. Due to the complexity, I refer the reader to Appendix D.2.











<details style="border: 0.5px black; border-style: dashed">
<summary> <b> How can we determine number of iterations using that result? </b> </summary>

Using the following inequality: $$ x \in (0,1) \implies 1-x \le e^{-x} $$ and by setting $$x = \eta c^{\frac{2(N-1)}{N}}$$ observe that the following holds:

$$\begin{equation}
L^1(t) \le L^1(0) exp(-\eta c^{\frac{2(N-1)}{N}}*t)
\end{equation}$$

Assume we want $$L^1(t) \le \epsilon$$ we observe that:

$$\begin{equation}
L^1(0) exp(-\eta c^{\frac{2(N-1)}{N}}*t) \le \epsilon \implies log(L^1(0)exp(-\eta c^{\frac{2(N-1)}{N}}*t)) \le log(\epsilon) 
\end{equation}$$

Since $$log(ab)=log(a)+log(b)$$ we can rearrange that equation to the following:

$$\begin{equation}
t \ge \frac{1}{\eta c^{\frac{2(N-1)}{N}}} log(\frac{L^1(0)}{\epsilon}) 
\end{equation}$$

</details>

However, it is crucial to note that the authors made a non-trivial assumption that made that proof feasible. They assumed that $$W_e$$ initialized such that it has **deficiency margin** $$c \gt 0$$ with respect to the target matrix, denoted as $$\Phi$$, which is the matrix that achieves zero loss, i.e. global minimum. Specifically, they defined *deficiency margin* in the following way:







$$\begin{equation}



||W-\Phi||_F \le \sigma_{min}(\Phi)-c



\end{equation}$$

Although it is difficult to see how this assumption can help, it is important to remember that they were trying to **limit the loss decrease**. They made it feasible by showing that the minimum singular value of the parameters, $$\sigma_{min}(W_e(t))$$, is lower bounded by some constant, namely **c**. Using that, they were able to obtain a non-trivial decrease. Again, the specific details can be found in Appendix D.2, but keep in mind the overall picture - they showed that if the minimal singular value is bounded away from zero, a non-trivial loss decrease occurs that implies a linear convergence rate.

## Goals and Assumptions Quick Recap ##

The goal of this paper is to prove that gradient descent converges at a linear rate to a global minimum for LNN. However, the proof was made feasible through the introduction of two assumptions: *balancedness* and *deficiency margin*. While balancedness might be empirically reasonable, deficiency margin is not a trivial assumption and in a way assumes that the initialization point is pretty **close to the target** (and therefore enjoys a linear convergence rate). One can argue that a more general scenario should not assume such a successful initialization, and wonder what convergence rate can be ensured without this assumption.







But even if one could mitigate the deficiency margin assumption, extending this result to a DNN with a **non-linear** activation is non-trivial. Luckily, researchers found that if we make some assumptions about the network's architecture (regarding width & depth), we can reason on the convergence of much more complex neural networks!






# Over-parameterized Neural Networks and Neural Tangent Kernel #

When researchers studying parameters trajectory analysis hit challenges generalizing the results shown above to more practical cases, a new method to handle the optimization problem in deep learning was needed. 







A recurring theme in research is observing phenomena at extremes. In recent years it is well-known that wider and deeper networks are achieving outstanding results. Therefore, one might wonder how an infinitely wide neural network performs? 







Fortunately, as we shall see now, investigating the effect of width led to convergence proofs that could eliminate strict assumptions on the initialization, such as the deficiency margin. Interestingly, those results follow in general the high-level recipe presented above for the trajectory approach. 















## Dynamics Definition ##



Previously, we had seen a definition of a dynamical system with the parameters $$W$$ as its underlying state. Let us now observe what happens when we define the state to be the **predictions**. The following relies heavily on the discussion section at [Du et al., 2018](https://arxiv.org/pdf/1810.02054.pdf).


Assume that we deal with a quadratic loss. i.e. $$L=\frac{1}{2}(y'-y)²$$, and we have a non-linear deep NN with N hidden layers. Also assume we are performing a regression task, and denote the output layer (which is a fully connected layer) as **a**. The network's prediction vector is defined as:

$$\begin{equation}
u_i = f(W,a,x_i) = a^T\sigma(W_N \cdot \cdot \cdot \sigma(W_1x_i))
\end{equation}$$

where $$\sigma$$ denotes some non-linear activation function, $$x_i$$ is some input and $$u_i$$ defined to be the prediction on that input. Also, let us continue with the continous analysis with gradient flow, and recall from eq. 2 (with a slight abuse of notation) that $$\dot W = \frac{dW(t)}{dt} = -\frac{\partial L(W(t))}{\partial W(t)}$$.

Using the chain rule and definition of partial derivatives we observe that:

$$\begin{equation}
-\frac{\partial L(W(t))}{\partial W(t)} = -\frac{\partial L(W(t))}{\partial u(t)}\frac{\partial u(t)}{\partial W(t)} = -\sum_{i \in M}\frac{\partial L_i(W(t))}{\partial u_i(t)}\frac{\partial u_i(t)}{\partial W(t)}
\end{equation}$$

Where $$L_i(\cdot)$$ denotes the loss for the input $$i \in M$$, for a dataset of M samples. Observe that the derivative of the loss to $$u_i(t)$$ is trivial and from the definition of the specific loss we are working with (quadratic loss), we get:

$$\begin{equation}
-\frac{\partial L(W(t))}{\partial W(t)} = \sum_{i \in M}(y-u_i(t))\frac{\partial u_i(t)}{\partial W(t)}
\end{equation}$$

With this information in hand, we can return to our current task - defining the **prediction dynamics**. Specifically, we want to discover how they change in time based on their current state. Using (once again...) the chain rule we note that:

$$\begin{equation}
\frac{du_i(t)}{dt} = \frac{\partial u_i(t)}{\partial W(t)}\frac{dW(t)}{dt}
\end{equation}$$

And here comes the magic moment! We have seen above how we can express that last term. Let us substitute it and arrange the indices (changing previous notation from i to j):

$$\begin{equation}
\frac{du_i(t)}{dt} = \frac{\partial u_i(t)}{\partial W(t)}\sum_{j \in M}(y-u_j(t))\frac{\partial u_j(t)}{\partial W(t)}
\end{equation}$$

If we rearrange the sums we get:

$$\begin{equation}
\frac{\partial u_i(t)}{\partial W(t)}\sum_{j \in M}(y-u_j(t))\frac{\partial u_j(t)}{\partial W(t)} = \sum_{j \in M}(y-u_j(t))\frac{\partial u_i(t)}{\partial W(t)}\frac{\partial u_j(t)}{\partial W(t)}
\end{equation}$$

And here we arrived at the cornerstone of the predictions trajectory analysis approach. Let us denote the multiplication of the derivatives in the following manner:

$$\begin{equation}
G_{ij}(t)=\frac{\partial u_i(t)}{\partial W(t)}\frac{\partial u_j(t)}{\partial W(t)}
\tag{9}
\end{equation}$$

And therefore the dynamics simplified to:

$$\begin{equation}
\frac{du_i(t)}{dt} = \sum_{j \in M}(y-u_j(t))G_{ij}(t)
\tag{10}
\end{equation}$$

And observe that we arrived at a definition of the dynamical system induced by defining the state as the prediction of the network. So far we did not require any assumption and thus this result holds in general. Observe, too, that the matrix **G** from eq. 9 is **random** (in a sense that it is dependent on the initialization) and **time-dependent** (changes in time, as the parameters change). This makes it very difficult to analyze and therefore researchers looked to simplify it by investigating the effect of increasing the network's width, as we will soon learn.   

> Note that these dynamics are referred to in different ways across the literature but are basically the same quantity. 
> The works that followed the Neural Tangent Kernel approach by [Jacot et al., 2018](https://arxiv.org/abs/1806.07572) refer to it as a **kernel** as we will soon try to understand why. Other works simply referred to it as the **Gram matrix** of the predictions derivatives. Anyhow, it is essentially the same dynamics.

As the goal of this blog post is to give you a high-level understanding of the trajectory analysis approach, we will now overview several exciting results that stem from the definition of the dynamics in eq. 10. These works investigated how precisely increasing the network's width enabled the prediction dynamics to be tractable for analysis.

<!-- , we will now do a quick overview on we will now cover the results of [Du et al., 2018](https://arxiv.org/pdf/1810.02054.pdf), that proved convergence for shallow (two layers) non-linear neural network, and then highlight more modern and stronger results for deep NNs. -->

## Convergence (at a linear rate) to Global Minimum ##

It is important to note that the dynamics of eq. 10 holds in general, and we did not introduce the effect of over-parameterization. Additionally, recall how the convergence to global minimum (and the rate at which it occurs) proof was built previously. With the **parameters dynamics** at hand, [Arora et al., 2018](https://arxiv.org/pdf/1810.02281.pdf) looked to lower bound the loss decrease to show a non-trivial decrease towards zero. However, they needed to set a non-trivial assumption (deficiency margin) on the initialization for it to occur, and we can interpret the assumption as requiring the initialization to be "close enough" to the target. 

An influential step towards mitigating these initialization assumptions was made by [Jacot et al., 2018](https://arxiv.org/pdf/1806.07572.pdf). They investigated what happens when we let the network be infinitely wide, and found an interesting equivalence between these infinitely-wide neural networks and kernel machines that are accompanied by the kernel we saw at eq. 9.  

<details style="border: 0.5px black; border-style: dashed" markdown="1">
<summary><b> A quick recap on kernel machines</b> </summary>
Despite the fact that kernel machines were one of the strongest tools out there for learning-based models for decades, they are (probably) not that well-known among deep learning practitioners nowadays. In some sense, it is not a surprise since in the recent decade neural networks emerged as a powerful and general solution for learning problems.

However, one can say that the theory behind kernel machines experiences a resonance through the research of deep learning theory, as we shall now see. But before that, let us do a (very) quick recap on kernel machines.

A kernel machine is an **instance-based learner**. Basically, it means that the kernel machine remembers all instances it has seen during training, and at test-time, it calculates **how similar the given input is to all previously seen inputs**. Using this similarity metric, the prediction for the given input is some weighted aggregation of the previously seen corresponding labels. 

In particular, for a given kernel (regressor) machine that is defined by a kernel **K**, the prediction on an unseen input **x**, given a dataset $$\{x_i,y_i\} _{i \in N}$$ can be defined as:

$$\begin{equation}
y = \sum_{i \in N}w_iy_iK(x,x_i) + b
\end{equation}$$

Note that $$w_i$$ and $$b$$ can be seen as the learned linear model that is determined by the learning algorithm. Therefore, kernel machines are relatively simple models. Their analysis is based on convex optimization and the theory behind them is well-founded. In a sense, the main "heavy-lifting" behind them is coming up with the kernel K that defines a suitable similarity measure between inputs that is efficient to compute. Fortunately, as we shall see next, neural networks can be seen as somehow implicitly learning the kernel function from data.

</details>

Without diving into the specific details, let us try to get an intuition to that equivalence. Let us begin with a simple case - a shallow (one-layer) neural network with infinitely many neurons in a single layer. In that case, one can intuitively observe that at initialization (t=0), the outputs $$ {u_i(0)}_{\forall i \in D}$$ from dataset D converges to a Gaussian with some known mean standard deviation. This is justified by the Central Limit Theorem (for our purposes, recall that it says that a sum of i.i.d random variables, under proper normalization, converges toward a Gaussian distribution). Assume that we have that proper normalization at hand, we can now add another layer to the network. We fix the first network's weights and apply that process again (taking the width to $$\infty$$ and finding the proper normalization). This can be done recursively and show that infinitely-wide neural networks are in correspondence with (a certain class) of Gaussian Processes. For a more formal introduction to the equivalence between neural networks and Gaussian processes, please refer to [Lee et al., 2019](https://arxiv.org/pdf/1902.06720.pdf).

But an equivalence to a Gaussian process is not enough. We still have one crucial step to make. The next step is to show that the infinitely-wide neural network can be well-approximated by a Taylor series centered around the initial parameters. This Taylor expansion is often referred to as a linearized model, and its dynamics have a closed-form solution (under the quadratic loss, for example). It means that we only need to compute $$u(0)$$, and $$G(0)$$ (outputs and kernel/Gram matrix in initialization) and use **simple ODEs to compute the outputs at time t**. 

<!-- In a sense, the fact that infinite-width network are equivalent to linearized models may suggest the following insight. Since there are infinite number of weights, it is sufficient that each one of them would change by only a vanishingly small amount for them to collectively provide the neccessary change for training.   -->

Since this result deals with infinitely-wide neural networks, many papers tried to discover more practical scenarios where over-parameterization is still beneficial to the theoretical analysis of gradient descent.

In particular, most of those papers tackle the following two steps to show equivalence between an over-parameterized neural network and some fixed kernel, as described in this [blog post](http://www.offconvex.org/2019/10/03/NTK/) by Wei Hu and Simon Du. The two steps are:

1. **Convergence at initialization** - As the major difficulty in analyzing the dynamics from eq. 10 is the time-dependency property of the matrix $$G(t)$$, the first step tries to eliminate this time-dependency. Most papers try to show that (under proper scaling) the time-varying kernel induced by the **parameters of the network at initialization**  converges to the deterministic kernel $$K_{NTK}$$.

2. **Stability during training** - Above fact that the time-varying kernel converges to the deterministic kernel, most papers conclude that the time-varying kernel is stable during training. This essentially means the weights remain close to their initialization, which is in some sense the $$K_{NTK}$$. It is not a trivial concept to grasp, and one might stick to the intuition provided in the above-mentioned blog post: "Intuitively, when the network is sufficiently wide, each weight only needs to move a tiny amount in order to have a non-negligible change in the network output."

As the specific details of each step are highly involved, we will only conclude here by highlighting interesting results. [Du. et al., 2018](https://arxiv.org/pdf/1810.02054.pdf) showed that a sufficiently wide two-layer (non-linear) shallow neural network converges at a linear rate. In follow-up work, [Du. et al., 2019](https://arxiv.org/pdf/1811.03804.pdf) generalized this analysis to more interesting architectures (deep fully-connected NNs, ResNets, and even CNNs) and was able to prove convergence at a linear rate as well. Note that those works also followed to some extent the high-level recipe presented in the beginning. They defined the predictions dynamics as we did above, and then analyzed at what conditions (what is the required width, depth, and learning rate) those dynamics would induce a convergence to the global minimum, and at what convergence rate.  

---

# Discussion and Conclusion #

As we tried to tackle the mysterious success of gradient descent, a relatively simple first-order optimization algorithm, on deep (and wide!) non-linear neural networks. As far as I know, the usage of the high-level recipe of the trajectory analysis is the most effective method for studying optimization in deep learning today.

This approach aims to model the dynamics of gradient descent and use them to prove convergence (with mild assumptions). However, we have seen that even studying "simple" deep **linear** neural networks is not easy at all. Fortunately, setting the object we analyze in time to be the predictions was a step that revealed interesting phenomena on non-linear over-parameterized NNs.

## On the Assumption of Gradient Flow ##

While the results are encouraging, it is relevant to note that throughout all this post we made one assumption without justifying it. We assumed that gradient flow approximates the discrete gradient descent in a way that allows us to study the first and make conclusions on the other. But one might ask to what extent does gradient descent with conventionally used learning rates is actually "close" to gradient flow? Fortunately, [Elkabatz. et al., 2021](https://arxiv.org/pdf/2107.06608.pdf) recently published a paper that studies this question and comes up with optimistic results that show the common learning rate for gradient descent is a reasonable approximation for gradient flow!

## First Order Approximation of Over-parameterized Neural Networks  ##
We observed that there is a connection between over-parameterized neural networks and linearized models. They are equivalent in the infinite width limit and are approximately equal for large over-parameterized networks. Those who describe this approximation as a linearized model have the following intuition: Since there are plenty of weights, even if each one changes only a tiny amount, they provide the necessary change for training. Therefore, it is assumed that this approximation holds since the change of each weight can be modeled linearly. 

However, as noted by [Arora et al., 2019](https://arxiv.org/pdf/1904.11955.pdf), the empirical performance of those linearized networks is observed to be outperformed by their corresponding practical neural networks. Interestingly, [Yu et al., 2020](https://openreview.net/pdf?id=rkllGyBFPH) studied that gap and analyzed higher-order approximations of neural networks (that are still governed by the Taylor expansion of the network).


<!-- ## Tight Connection to Generalization ##
Since this post aims to highlight the progress researchers have made in recent years, we should point out that **the optimization question should not be viewed in isolation**. As it is becoming more evident, there is a strict connection between the optimization algorithm to the generalization performance of the optimized neural network. Therefore, the lessons we learned today that tackle optimization can help tackle the even more mysterious generalization phenomena of deep learning. We hope you will use those tools to learn more about it yourself.  -->

---
Well, that was a long one. This is a really interesting field for me and I hope that this post will be helpful for others interested in this field. Please let me know about mistakes I made in my post and suggestions for corrections :)