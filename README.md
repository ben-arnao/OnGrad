# OnGrad (online gradient estimation)
A derivative free reinforcement learning algorithm

# Motivation for an alternative

State of the art reinforcement learning methods like PPO or SAC can leave a lot to be desired when used on complex problems to achieve competitive performance. There are a few big shortcomings...

1) A differentiable loss function is usually required to model reward distributions. Many real life problems either are not differentiable or if there does exist a differentiable loss function for the final episode performance with respect to model parameters, the loss function actually used is probably suboptimal at best. That is to say, that a decrease in loss does not always translate to an increase in final episode performance in a direct manner. One can expect a loose correlation, but not much more. This might be fine if we could expect zero error predictions (more or less using a lookup table to know the exact value distribution for every time step), but this is not reality, and we should always be expecting only an approximation. This sort of disconnect can be problematic for many reasons... mainly that we are not actually optimizing performance directly but rather some auxiliary function that only has a correlation with performance.

2) These methods usually require the practitioner to guess at or arrive at by trial and error a good value for a single static time horizon value (alpha). This may be fine if we want something decent or good enough, but to achieve high-end competitive performance, a model will more than likely need to consider a wide variety of time horizons at varying points of play.

3) They can rely on tedious exploration strategies and backpropagation of reward which require many samples/episodes.

4) They can be overly complex and easy to break. Some even require specialized methodologies to deal with a particular use case.

One such method that was proposed to solve some of these shortcomings is Natural Evolutionary Strategies (NES) https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf. NES takes a few big steps in the right direction but ultimately falls short in a few key areas. NES is very slow, requiring in some cases tens and thousands of samples to estimate the gradient for just a single step. It also suffers from instability as optimization gets into tougher and higher score areas, preventing optimization from reaching into higher maxima.

# OnGrad explained

OnGrad incorporates a novel way of calculating gradients to improve sample efficiency and and also get quality estimates.

* Noise in the weights is scored similar to NES. We use the score of the positive and negative noise to determine the sign for a single sample. We accumulate many samples to get an average of which direction the weights should move in. The "stronger" the signal in one direction over the other is, the bigger the gradient and therefore the bigger step we will take. Conversely if the direction is indecisive and the average sign is near 0, the gradient will be smaller and a smaller step will be taken.

There are a few nuances..

1) Unlike the original NES paper where the estimate is recalculated from scratch every step, OnGrad assumes the gradient for the next step will be more similar to the last iteration rather than dissimilar, and thus we use the last estimate as a base for the next estimate.

Probably the biggest improvement as far as sample efficiency is concerned- only the amount of samples needed to update the estimate are calculated, drastically improving sample efficiency. Previously, a static amount of samples was calculated for every step without really knowing if this number was too little for a good estimate or too many such that extra samples did not improve the quality of the estimate.

OnGrad solves this issue by tracking the upper and lower bounds of the gradient estimate moving average. We keep accumulating samples and adding to the estimate, until the estimate is deemed “stationary enough”. This is defined by tracking the percent of estimates that remain within a percentage of the current high and low bounds for a single estimation step. If this percent goes above a threshold, we say that the estimate is good enough and take the step. This means that for some steps where the true gradient does not change very much, not many samples need to be calculated. For steps however where there is a lot of change, we can also dynamically calculate more samples to ensure we get the same quality estimate.

2) The other nuance of OnGrad's gradient estimation is that momentum is scaled by the magnitude of the random noise. Smaller noise will result in a larger momentum, and therefore a smaller impact to the estimate. Conversely, larger noise means a smaller momentum, and bigger adjustment to the estimate. This is primary to ensure very small and likely insignificant noise values, do not affect the estimate as much as bigger noise will likely have an impact on the estimate.

* Since we operate directly on the gradient of final episode score (what we really care about), we eliminate all of the complications and messiness that come with trying to model reward distribution at a per time-step/action level.

* Actual step is always scaled so that the mean of the step for every weight always equals the current LR, ensuring that general step size and gradient magnitude is decoupled.
 
Despite the notion that we need tens and thousands of samples to estimate gradients properly, I've found that estimating gradients this way only requires a fraction of the samples to obtain a quality gradient estimate that we can use to traverse the score space well into very high optima.

* We also make step size a factor of noise size, as our gradient estimate's scope is bounded by the size of the noise. That is to say that for example, step size might start out such that the average step magnitude is the standard deviation of the noise. This is another way to decouple parameters and ensure that one thinks of step size as a factor of the noise size (because gradient is only estimated within the scope of the noise)

* Likewise we make weight decay a factor of the step taken as well. Such that every iteration, the percent that weights are decayed by, is scaled by the magnitude of the step size.

* Step is clipped by a factor of noise. This is done to ensure the step does not exceed the area that we are actually estimating the gradient for.

* A simple but critical component… step size (and optionally noise size) is reduced as we go on

The end result is a RIL algorithm that from my experience tackles all of these issues and does it in a simple and more intuitive way. Please try out OnGrad for yourself and please share the results!

# Usage

The user needs to provide 4 parameters...

```get_model_params``` which takes a model, and outputs a flattened list of model parameters

```set_model_params``` which takes a model and a flattened list of model parameters, and sets the model with the provided parameters

```get_episode_score``` this is a function which accepts a model as a parameter, and output the score of this model (ex. final episode score)

```model``` this is the supplied model (can be tensorflow, pytorch, etc.)


