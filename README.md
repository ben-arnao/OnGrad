# OnGrad (online gradient estimation)
A derivative free reinforcement learning algo

State of the art reinforcement learning methods like PPO or SAC leave a lot to be desired when used on complex problems to achieve competitive performance. There are a few big shortcomings.

1) A differentiable loss function is required in some capacity in most cases. Many real life problems either don't have one at all, or the function that ends up being used is suboptimal at best. That is to say, that a decrease in loss does not always translate to an increase in final epsiode performance in a straightforward manner. One can expect a loose correlation, but not much more. This is fine, if in a perfect world we can expect to decrease loss indefinitely until we reach 0. That is not reality, and if there is ever a scenario in which a higher loss can mean better final episode performance, this can be problematic on a number of different levels for obvious reasons.

2) They usually require the practitioner to guess at or arrive at by trial and error a good value for a single static time horizon value (alpha). This may be fine if we want something decent, but to achieve high end competitive performance on some tasks, a model will more than likely need to consider a wide variety of time horizons at varying points of play.

3) They also can rely on tedious exploration strategies and backpropagation of reward which require many samples.

4) These methods can be overly complex. Some even require specialized methodologies to deal with a particular use case.

Reinforcement learning is at reckoning with Occam's razor.

One such method that was proposed to solve some of these shortcoming is Natural Evolutionary Strategies (NES). NES takes a few big steps in the right direction but ultimately falls short in a few key areas. NES is very slow requiring tens and thousands of samples to estimate the gradient for just a single step. It also suffers from instability as optimization gets into tougher higher score areas, preventing optimization from reaching into higher maxima.

OnGrad takes a step back from all the hype and incorporates a novel way of calculating gradient that carries over from step to step. We accumulate sample scores in an additive manner. When it comes to gradient, all we care about is relative magnitude and scale is irrelevant. One can also assume gradient smoothly transforms throughout optimization
, such that having a base from the previous step, can speed up estimate saturation.

We start out accumulating more samples per step when LR is high and gradient changes more rapidly, and scale it down as LR gets lower and gradient doesn't change too much. Therefore we don't need as many estimates to get a fresh and accurate estimate from step to step.

We also decay the gradient estimate each step, to ensure we get fresh estimates and mainly to ensure that existing gradient estimates don't stagnate at an old value when the current gradient is close to 0.

I've found that estimating gradient this way only requires a fraction of the samples to obtain a gradient estimate good enough that we can traverse the score space well into very high optima.

We also make step size as a factor of noise size, as our gradient estimate's scope is bounded with the size of the noise. Likewise we make weight decay a factor of the step taken as well. Steps are then clipped as a factor of noise so that it doesn't not exceed the actual noise magnitude by too much. Although one can see steps higher than the value of the noise as confidence, we want to find a middle ground where we take big steps if there is a high enough confidence (big enough gradient), but do not bite off more than we can true and risk overshooting into suboptimal landscape.

As mentioned previously, one the main issues I found with NES is that it would be wildly unstable as you reached better optima and the algorithm was unable to escape such a pattern and ascend into better optima.

OnGrad attempts to solve this issue in two ways...

1) We add extra "recalibration" estimate samples after our score drops. This scale with the % score drop, such that the bigger the drop, the more samples accumulated for next gradient estimate. One wants to be cautious here, as sometimes a drop is normal onto better optima, so setting the recalibration samples too high, can cause optimization to lose it's momentum.

2) A tried and true tactic, we also lower LR as we go on. We also increase the LR reduce patience each time the LR drops, allowing optimization to climb back into better optima with smaller steps. I've found the gradually reducing LR (by a factor of 2 for example) works the best here.

The end result is a RIL algorithm that from my experience tackles all of these issues and does it in a simple and more intuitive way. Please try out OnGrad for yourself and please share the results!
