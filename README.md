# OnGrad (online gradient estimation)
A derivative free reinforcement learning algorithm

# Motivation for an alternative

State of the art reinforcement learning methods like PPO or SAC can leave a lot to be desired when used on complex problems to achieve competitive performance. There are a few big shortcomings...

1) A differentiable loss function is usually required to model reward distributions. Many real life problems either are not differentiable or if there does exist a differentiable loss function for the final episode performance with respect to model parameters, the loss function actually used is probably suboptimal at best. That is to say, that a decrease in loss does not always translate to an increase in final episode performance in a direct manner. One can expect a loose correlation, but not much more. This might be fine if we could expect zero error predictions (more or less using a lookup table to know the exact value distribution for every time step), but this is never the case in real life scenarios, and we should always be expecting only an approximation. This sort of disconnect can be problematic for many reasons because we are not actually optimizing performance directly, but rather some auxiliary function that only has a correlation with performance.

2) These methods usually require the practitioner to guess at or arrive at by trial and error a good value for a single static time horizon value (alpha). This may be fine if we want something decent or good enough, but to achieve high-end competitive performance a model will more than likely need to consider a wide variety of time horizons at varying points of play.

3) They can rely on tedious exploration strategies and backpropagation of reward which require many samples/episodes.

4) They can be overly complex and easy to break. Some even require specialized methodologies to deal with particular use cases.

One such method that was proposed to solve some of these shortcomings is Natural Evolutionary Strategies (NES) https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf. NES takes a few big steps in the right direction but ultimately falls short in a few key areas. NES is very slow, requiring in some cases tens and thousands of samples to estimate the gradient for just a single step. It also suffers from instability as optimization gets into tougher and higher score areas, preventing optimization from reaching into higher maxima.

# OnGrad explained

OnGrad introduces a novel way of estimating gradients to improve sample efficiency while still retaining the quality needed ascend into very high score spaces

Noise in the weights is scored similar to NES... The score of the positive and negative noise is used to determine sign for a single sample. We accumulate many samples via a momentum-based moving average to estimate the gradient for a single step. The results average essentially represents a form of confidence. A value closer to 1 or -1 indicates a greater confidence that this direction is better than the other, and therefore a bigger step is taken.

Since we operate directly on the gradient of the final episode score (what we really care about), we eliminate all of the complications and messiness that come with trying to model reward distribution at a per time-step/action level.

Unlike the original NES paper where the estimate is recalculated from scratch every step, OnGrad assumes the gradient for the next step will be more similar to the last iteration rather than dissimilar, and thus we use the last estimate as a base for the next step to greatly reduce the amount of overall samples needed.

Only the amount of samples needed to update the estimate are calculated, drastically improving sample efficiency. Previously, a static amount of samples was calculated for every step without really knowing if this number was too little for a good estimate or too many such that extra samples did not reasonably improve the quality of the estimate.

OnGrad solves this issue by tracking the upper and lower bounds of the gradient estimate moving averages. We keep accumulating samples and adding to the estimate, until the estimate is deemed “stationary enough”. This is defined by tracking the percent of estimates that do not produce a new high or low bound. If this percentage goes above a threshold, we say that our estimates are good enough and we take the step. This means that for some steps where the true gradient does not change very much, not many samples need to be calculated. For steps however where there is a lot of change, we can also dynamically calculate more samples to ensure we get the same quality of estimate.

One experiment that can be performed to further analyze the properties of estimating gradients in this manner, is to initialize two sets of gradient estimate containers to random values. We then will accumulate samples and add to the moving average in parallel, and compare the convergence between the two. We calculate convergence by taking the mean absolute difference of the estimates. The smaller the difference, the closer the estimates are to each other and therefore to the true value/gradient, and the more accurate and stable the estimate is.

Here are the following plots for an experiment using a noise stddev of 0.01, estimating for a single step.

![mom 0.9](https://github.com/ben-arnao/OnGrad/blob/main/images/test_0.9.png?raw=true)
![mom 0.99](https://github.com/ben-arnao/OnGrad/blob/main/images/test_0.99.png?raw=true)
![mom 0.999](https://github.com/ben-arnao/OnGrad/blob/main/images/test_0.999.png?raw=true)

We can discern a few important pieces of information from this experiment...

1) As expected, a higher momentum produces a better estimate and likewise requires more samples.

2) We can see that for a momentum of 0.9, the quality of the estimate levels off at around 90%. This is a good point to say that we have reached a point of diminishing returns, where it takes a lot of samples to improve the quality a little bit, if at all. We can also see that with a higher momentum, the threshold to reach this point becomes higher as well.

3) A completely random set of estimates will on average have a difference factor of 1.0. We can see that with a momentum of 0.9 the best achievable estimate will have an error of 0.25, a momentum of 0.99 results in a BAE of 0.075, and finally a momentum of 0.999 looks to have a trajectory to reach a BAE of below 0.01.

This gives us some useful information that will help us setting these values (obviously these are very dependent on the users goals and the experiment). For momentum, one will probably want to keep this value between 0.99 and 0.999. For threshold, a value between 0.9 and 0.95 seems best.

It is important to keep in mind that for most problems, an estimate that is 100% accurate is not needed. From my own experiments I have found that a highly accurate but slow estimate causes the exact same score improvement plots, as an estimate that is fairly accurate but much faster. For most problems I have been able to get into very high score spaces with values that only required less than 100 samples per step.

A few other features to keep in mind

* Step is calculated via gradient_estimate * noise_stddev * step_size_factor.

* Noise (therefore step size as described above) is reduced when the score plateaus.

* Weight decay is a factor of the step taken as well. Such that every iteration, the percent that weights are decayed by, is scaled by the magnitude so that as step size is reduced during training, the magnitude of weight decay is reduced proportionally.

Please try out OnGrad for yourself and please share the results!

# Usage

The user needs to provide 4 parameters...

```get_model_params``` which takes a model, and outputs a flattened list of model parameters

```set_model_params``` which takes a model and a flattened list of model parameters, and sets the model with the provided parameters

```get_episode_score``` this is a function which accepts a model as a parameter, and output the score of this model (ex. final episode score)

```model``` this is the supplied model (can be tensorflow, pytorch, etc.)

```init_routine``` this is a weight initialization routine to ensure we start with a model where noise is able to produce varying scores. One example might be to pre-train the model on a set of random actions to simulate an epsilon greedy policy.
