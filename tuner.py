import numpy as np
from matplotlib import pyplot


def train(

        # <><><> functions users needs to supply for their environment
        get_model_params,
        set_model_params,
        get_episode_score,  # input is model, return a positive float for score
        model,  # supplied model, can be a pytorch or tensorflow model

        # <><><> training params

        # reduce lr and end training patience
        lr_patience=25,
        end_patience=50,

        grad_noise=2e-2,  # noise in weights used to estimate gradient. this is very problem dependant
        # default should be good enough in most cases

        weight_decay_coeff=0.1,  # weight decay = mean absolute step size * 'weight_decay_coeff'

        step_norm_factor=1,  # step size is always a factor of the noise size

        step_clip_factor=3,  # we do not want to take steps that exceed the size of noise used for grad estimate
        # recommended 1-5

        recaliberation_factor=100,  # coeff that determines amount of recalibration iters after score drops

        momentum=0.9,  # momentum for gradient estimate

        base_est_iters=1  # base amount of samples to add to gradient estimate per step

):
    # get baseline
    best_score = get_episode_score(model)

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
        if pos_rew != neg_rew:
            score_diff = abs(pos_rew - neg_rew)
            grad = np.where(pos_rew > neg_rew, grad + v * score_diff, grad - v * score_diff)

        # decay gradient est to fresh estimate over time
        grad *= momentum

        return grad

    while True:

        # base samples each iteration used to estimate gradient
        for _ in range(base_est_iters):
            grad = add_sample_to_grad_estimate(grad)

        # <<<< calculate step >>>>
        # rescale grad to mean of 1 because with grad we only care about values relative to another.
        # if we don't rescale, step_size can be affected by the scale change of score, which is irrelevant to gradient
        # then make step size a factor of noise used for estimating gradient
        grad_rescaled = grad / np.mean(np.abs(grad))
        step = grad_rescaled * ((grad_noise * step_norm_factor) / np.mean(np.abs(grad_rescaled)))

        # clip step size to try and not exceed the noise used for gradient estimation
        step = np.clip(step, -grad_noise * step_clip_factor, grad_noise * step_clip_factor)

        # calculate average step magnitude for use in determining weight decay factor
        step_mag = np.mean(np.absolute(step))

        # get step score
        new_score = get_step_score(model, step)

        # take step
        set_model_params(model, get_model_params(model) + step)

        # decay weights where weight decay value is modified by the step magnitude
        set_model_params(model, get_model_params(model) * (1 - weight_decay_coeff * step_mag))

        # calculate num recalibration samples based on score drop percentage
        if new_score < score:

            if score < 0 or new_score < 0:
                raise NotImplementedError('Does not currently handle negative scores. consider adding a baseline and'
                                          'clipping anything under to 0')

            drop_percent = abs(new_score / score - 1)
            recalibration_samples = 1 + int(round(drop_percent * recaliberation_factor))
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
                # lower the LR the more patience we should have to end training
                end_patience += 5
                step_norm_factor /= np.sqrt(10)
                lr_reduce_wait = 0

            if iters_since_last_best >= end_patience:
                return score_history
