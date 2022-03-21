# OnGrad (online gradient estimation)
A derivative-free reinforcement learning algorithm

# Motivation for an alternative

State of the art reinforcement learning methods like PPO or SAC can leave a lot to be desired when used on complex problems to achieve competitive performance. There are a few big shortcomings:

1) A differentiable loss function is usually required to model reward distributions. However, many real life problems are either not differentiable or, if there does exist a differentiable loss function for the final episode performance with respect to model parameters, the loss function actually used is probably suboptimal at best. That is to say, a decrease in loss does not always translate to an increase in final episode performance in a direct manner. One can expect a loose correlation, but not much more. This might be good enough if we could eventually expect zero error predictions (more or less using a lookup table to know the exact value distribution for every time step), but this is never the case in real life scenarios. Neural networks are used not only to approximate these otherwise un-feasibly large reward tables, but more importantly to provide a means to predict on potentially unseen inputs. This sort of disconnect with the loss function used and actual final episode performance can be problematic for many reasons. Training will likely be inefficient and final model performance can be subpar.

2) These methods usually require the practitioner to guess at or arrive at by trial and error a good value for the single static time horizon value (alpha). For some problems this may be fine if we want something decent or good enough, but to achieve high-end competitive performance a model will more than likely need to consider a wide variety of time horizons at varying points of play.

3) They can rely on tedious exploration strategies and backpropagation of reward which require many samples/episodes.

4) They can be overly complex and easy to break. Some even require specialized methodologies to deal with particular use cases.

One such alternative method that was proposed to solve some of these shortcomings is Natural Evolutionary Strategies (NES) https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf. NES takes a few big steps in the right direction but ultimately falls short in a few key areas. 1) NES is very slow, requiring in some cases tens and thousands of samples to estimate the gradient for just a single step. 2) It also suffers from instability as optimization gets into tougher and higher score areas, preventing optimization from reaching into higher maxima.

# OnGrad explained

OnGrad introduces a novel way of estimating gradients to improve sample efficiency while still retaining the quality needed ascend into very high score spaces

Noise in the weights is scored similar to NES... The score of the positive and negative noise is used to determine sign for a single sample. We accumulate many samples via a momentum-based moving average to estimate the gradient for a single step. This average essentially represents a form of confidence. A value closer to 1 or -1 indicates a greater confidence that this direction is better than the other, and therefore a bigger step is taken relative to the other steps.

Since we operate directly on the gradient of the final episode score (what we really care about), we eliminate all of the complications and messiness that come with trying to model reward distribution at a per time-step/action level.

Unlike the original NES paper where the estimate is recalculated from scratch every step, OnGrad assumes the gradient for the next step will be more similar to the last iteration rather than dissimilar, and thus we use the last estimate as a base for the next step to greatly reduce the amount of overall samples needed.

When tuned correctly, OnGrad only calculates enough samples to "re-saturate" the estimate for the next step, drastically improving sample efficiency. Previously, a static amount of samples was calculated for every step without really knowing if this number was too little for a good estimate or too many such that extra samples did not reasonably improve the quality of the estimate.

OnGrad solves this issue by tracking the upper and lower bounds of the gradient estimate moving averages. We keep accumulating samples and adding to the estimate, until the estimate is deemed “stationary enough”. This is defined by tracking the percent of estimates that do not produce a new high or low bound. If this percentage goes above a threshold, we say that our estimates are good enough and we take the step. This means that for some steps where the true gradient does not change very much, not many samples need to be calculated. For steps however where there is a high degree of change, the algorithm will dynamically calculate more samples to ensure we get the same quality of estimate.

Please try out OnGrad for yourself and share the results!

# Usage

The user needs to provide the following parameters...

```get_model_params``` a function which takes a model, and outputs a flattened list of model parameters

```set_model_params``` a function which takes a model and a flattened list of model parameters, and sets the model with the provided parameters

```get_episode_score``` a function which accepts a model as a parameter, and outputs the score of this model (ex. final episode score)

```model``` this is the supplied model (can be tensorflow, pytorch, etc.)

```init_routine``` this function is a weight initialization routine to ensure we start with a model where noise is able to produce varying scores. One example might be to pre-train the model on a set of random actions to simulate an epsilon greedy policy.
