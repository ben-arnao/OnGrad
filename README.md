# OnGrad
A derivative free reinforcement learning algo

Value-function based reinforcement learning algorithms have fallen by the wayside in recent years for a number of reasons. There is much literature that explains why in depth, but ultimately they do not do well in many real-life environments where reward can be sparse and the model needs to consider rewards at varying time horizons for many different pathways that can be taken.

One such attempt at forcing the model to focus on more relevant information is the advantage function, where we now try to predict the advantage of one action over another. As opposed to wasting a model's focus and energy trying to predict the value itself, all we need to know is how much better action A is than action B.

Policy gradient methods operate directly on episode score, which cuts out the middle man and ultimately increases model performance by a substantial amount.

There are numerous advantages of policy gradient methods and more importantly, not having to define a derivative for your problem. For most real life problems, actions can be sparse and have complex relationships with reward distributions across time varying time horizons. For most problems, it is not possible to have a well defined loss function associated with this type of environment, and for most, the only formal goal is that we want to maximize episode score.

What this all means is that trying model these dynamics becomes overly complex and leaves a lot of room for humans to not pick correct hyper parameters. It also allows for us to find differentiable losses that are not fully representative of our final goal, maximum episode score. This becomes even more true when we consider that one cannot expect a fully fitted model, only partially fitted. Where a model might decide to fit certain "easier" areas better than others, and given the complex nature of RIL environments, this might completely skew the idea that a better fitted model equals a higher performing model.

Many of today's methods also rely on tedious exploration and backpropagation of reward which require many samples.

Reinforcement learning is at reckoning with Occam's razor.

One such method that was proposed to solve some of these shortcoming is Natural Evolutionary Strategies (NES). NES takes a few big steps in the right direction but ultimately falls short in a few key areas. NES is very slow requiring tens and thousands of samples to calculate gradient for just a single step. It also suffers from instability as optimization gets into tougher higher score areas, preventing optimization from reaching into higher maxima.

OnGrad takes a step back from all the hype and incorporates a novel way of calculating gradient that carries over from step to step. We accumulate sample scores in an additive manner. When it comes to gradient, all we care about is relative magnitude and scale is irrelevant. One can also assume gradient smoothly transforms, such that having a base from the previous step, can speed up estimate saturation.

We start out accumulating more samples per step when LR is high and gradient changes more rapidly, and scale it down as LR gets lower and gradient doesn't change too much. Therefore we don't need as many estimates to get a fresh and accurate estimate from step to step.

We also decay the gradient estimate each step, to ensure we get fresh estimates and mainly to ensure that existing gradient estimates don't stagnate at an old value when the current gradient is close to 0.

I've found that estimating gradient this way only requires a fraction of the samples to obtain a gradient estimate good enough that we can traverse the score space well into very high optima.

We also make step size as a factor of noise size, as our gradient estimate's scope is bounded with the size of the noise. Likewise we make weight decay a factor of the step taken as well. Steps are then clipped as a factor of noise so that it doesn't not exceed the actual noise magnitude by too much. Although one can see steps higher than the value of the noise as confidence, we want to find a middle ground where we take big steps if there is a high enough confidence (big enough gradient), but do not bite off more than we can true and risk overshooting into suboptimal landscape.

As mentioned previously, one the main issues I found with NES is that it would be wildly unstable as you reached better optima and the algorithm was unable to escape such a pattern and ascend into better optima.

OnGrad attempts to solve this issue in two ways...

1) We add extra "recalibration" estimate samples after our score drops. This scale with the % score drop, such that the bigger the drop, the more samples accumulated for next gradient estimate. One wants to be cautious here, as sometimes a drop is normal onto better optima, so setting the recalibration samples too high, can cause optimization to lose it's momentum.

3) A tried and true tactic, we also lower LR as we go on. We also increase the LR reduce patience each time the LR drops, allowing optimization to climb back into better optima with smaller steps. I've found the gradually reducing LR (by a factor of 2 for example) works the best here.
