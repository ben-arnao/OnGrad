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

OnGrad incorporates a novel way of calculating gradients. Noise in the weights is scored similar to NES, but this time we calculate the percentage advantage between postive and negative noise. We use this, in combination with the magnitude of the per-parameter noise, to calculate a single sample to be added to our running estimate of the gradient.

Since we operate directly on the gradient of final epsiode score (what we really care about), we eliminate all of the complications and messiness that come with trying to model reward distribution at a per time-step/action level.

We can estimate the gradient in an additve manner because when it comes to gradient, all we care about is relative magnitude, the scale is irrelevant. Our step is always scaled so that the mean of the step for every weight always equals the current LR, ensuring that general step size and gradient magnitude is decoupled.

This estimate is retained from step to step, eliminating the need to estimate the gradient from scratch each step. When the step size is bigger more samples are used in the estimate since it can be assumed that gradient changes more rapidly in comparison to smaller steps, where we do not need as many samples per step.

The gradient estimate is decayed (by a factor of step size), to ensure we get fresh estimates but also to ensure that existing gradient estimates don't stagnate at an old value when the current gradient is close to 0.

Despite the notion that we need tens and thousands of samples to estimate gradients properly, I've found that estimating gradients this way only requires a fraction of the samples to obtain a gradient estimate good enough that we can traverse the score space well into very high optima.

We also make step size a factor of noise size, as our gradient estimate's scope is bounded by the size of the noise. That is to say that for example, step size might start out such that the average step magnitude is the standard deviation of the noise. This is another way to decouple parameters and ensure that one thinks of step size as a factor of the noise size (because gradient is only estimated within the scope of the noise)

Likewise we make weight decay a factor of the step taken as well. Such that every iteration, the percent that weights are decayed by, is scaled by the magnitude of the step size.

Lastly, steps are then clipped as a factor of noise as well to not exceed the actual noise magnitude by too much.

As mentioned previously, one of the main issues I found with NES is that it would be wildly unstable as you reached better optima and the algorithm was unable to escape such patterns and ascend into better optima.

OnGrad attempts to solve this issue in two ways...

1) We add extra "recalibration" samples to our estimate after our score drops. This scales with the percent score drop, such that the bigger the drop, the more samples accumulated for the next gradient estimate. One wants to be cautious here, as sometimes a drop is normal as we traverse into better optima, so setting the recalibration samples too high can cause optimization to lose it's momentum and also cause unnecessary computations.

2) A tried and true tactic, we also lower step size as we go on. However, we also increase the patience used for reducing step size each time the step size is reduced, allowing more steps for optimization to climb back into better optima with smaller steps. I've found that gradually reducing LR (by a factor of 2 for example) works the best here and that increasing the LR patience with each LR reduction is critical to the success of the algorithm.

The end result is a RIL algorithm that from my experience tackles all of these issues and does it in a simple and more intuitive way. Please try out OnGrad for yourself and please share the results!

# Usage

The user needs to provide 4 parameters...

```get_model_params``` which takes a model, and outputs a flattened list of model parameters

```set_model_params``` which takes a model and a flattened list of model parameters, and sets the model with the provided parameters

```get_episode_score``` this is a function which accepts a model as a parameter, and output the score of this model (ex. final episode score)

```model``` this is the supplied model (can be tensorflow, pytorch, etc.)
