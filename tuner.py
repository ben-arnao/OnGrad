import numpy as np
from matplotlib import pyplot


def train(

        # <><><> functions users needs to supply for their environment
        get_model_params,
        set_model_params,
        get_episode_score,
        model,  # supplied model, can be a pytorch or tensorflow model

        # <><><> training params

        # reduce lr and end training patience
        lr_patience=25,
        end_patience=100,

        grad_noise=2e-2,  # noise in weights used to estimate gradient. this is very problem dependant
        # but the default should be good enough in most cases

        weight_decay_coeff=0.1,  # weight decay = mean absolute step size * 'weight_decay_coeff'

        lr=1e-1,  # different than standard LR since grad is calculated differently. start out high
        # default value probably fine in most cases

        step_clip_factor=5,  # we do not want to take steps that exceed the size of noise used for grad estimate
        # recommended 3-10

        score_delta_warmup_scale=100,  # coeff that determines amount of recalibration iters after score drops
        # use a higher value if you see that your model is not recovering well after drops

        grad_decay=0.1,  # higher value == more adaptive estimate, lower value == more stable and slow moving

):
    # get baselines
    score = get_episode_score(model)
    best_score = score

    # setup score history tracking
    score_history = []

    # get param count
    num_params = len(get_model_params(model))

    # gradient estimate
    grad = np.zeros(num_params, dtype=np.float32)

    # prepare vars
    iters_since_last_best = 0
    lr_reduce_wait = 0

    # function to calculate the score of a step (but do not actually commit the step to the model)
    def get_step_score(model, step):
        theta = get_model_params(model)
        set_model_params(model, theta + step)
        score = get_episode_score(model)
        set_model_params(model, theta)
        return score

    def add_sample_to_grad_estimate(grad):
        # generate noise
        v = np.random.normal(scale=grad_noise, size=num_params)

        # calculate negative/positive noise scores
        pos_rew = get_step_score(model, v)
        neg_rew = get_step_score(model, -v)

        # keep running total for gradient estimate
        if pos_rew != neg_rew and pos_rew != 0 and neg_rew != 0:
            score_diff = abs(pos_rew - neg_rew)
            grad = np.where(pos_rew > neg_rew, grad + v * score_diff, grad - v * score_diff)

        # make scale independent (with gradients, all we care about relative value), scale is nothing more than LR
        grad = grad / np.mean(np.abs(grad))

        # decay gradient est
        grad *= 1 - grad_decay

        return grad

    while True:
        # always add at least one sample to estimate each iteration
        grad = add_sample_to_grad_estimate(grad)

        # calculate step
        step = grad * lr

        # clip step size to try and not exceed the noise used for gradient estimation
        step = np.clip(step, -grad_noise * step_clip_factor, grad_noise * step_clip_factor)

        # calculate average step magnitude for use in determining weight decay factor
        step_mag = np.mean(np.absolute(step))

        # get step score
        new_score = get_step_score(model, step)

        # take step
        set_model_params(model, get_model_params(model) + step)

        # decay weights where weight decay value is modified by the step magnitude
        # (this is why weight decay should be a higher value by default)
        set_model_params(model, get_model_params(model) * (1 - weight_decay_coeff * step_mag))

        # If score drops, calculate additional samples for gradient estimate.
        # The larger the drop, the more samples will be calculated and the more
        # accurate and "up to date" the estimate will be.
        if new_score < score:
            
            if score < 0 or new_score < 0:
                raise NotImplementedError('Does not currently handle negative scores. consider adding a baseline and'
                                          'clipping anything under to 0')
            drop_percent = abs(new_score / score - 1)

            # calculate num samples based on drop percentage, momentum (higher mom, more samples), and
            # a user-supplied coefficient
            recalibration_samples = 1 + int(round(drop_percent * score_delta_warmup_scale))
            for _ in range(recalibration_samples):
                grad = add_sample_to_grad_estimate(grad)

        # set score after calculating score delta
        score = new_score
        score_history.append(score)

        # plot train/test history
        pyplot.clf()
        pyplot.plot(score_history, linewidth=0.35)
        pyplot.savefig('score history.png')

        if score > best_score:
            best_score = score
            iters_since_last_best = 0
            lr_reduce_wait = 0
        else:
            iters_since_last_best += 1
            lr_reduce_wait += 1

            # reduce lr/grad decay
            if lr_reduce_wait >= lr_patience:
                # since we take less recalibration steps and lr gets lower it will take longer for estimate to saturate
                lr_patience *= 1.5
                lr /= np.sqrt(10)
                lr_reduce_wait = 0

            if iters_since_last_best >= end_patience:
                return score_history
