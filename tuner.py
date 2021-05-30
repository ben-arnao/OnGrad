import numpy as np
from matplotlib import pyplot


def train(

        # <><><> functions users needs to supply for their environment
        get_model_params,
        set_model_params,
        get_episode_score,  # input is model, return a positive float for score
        model,  # supplied model, can be a any type of model as long as you define get/set params as well as ep score

        # <><><> training params

        # reduce lr and end training vars
        patience=20,
        lr_reduce_factor=2,
        patience_inc=10,

        noise=1e-2,  # noise in weights used to estimate gradient. this is very problem dependant
        # default should be good enough in most cases

        weight_decay_coeff=0.1,  # weight decay = mean absolute step size * 'weight_decay_coeff'

        init_lr=3,  # step size is a factor of noise

        step_clip_factor=10,  # a way to keep steps from being too much larger than estimation noise

        recaliberation_factor=25,  # coeff that determines amount of recalibration iters after score drops

        momentum=0.9,  # momentum for gradient estimate

        base_est_iters=3  # base amount of samples to add to gradient estimate per step

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
    lr = init_lr

    # function to calculate the score of a step (but do not actually commit the step to the model)
    def get_step_score(model, step):
        theta = get_model_params(model)
        set_model_params(model, theta + step)
        score = get_episode_score(model)
        set_model_params(model, theta)
        
        if score < 0:
            raise NotImplementedError('Does not currently handle negative scores. consider adding a baseline and'
                                      'clipping anything still negative to 0')
        return score

    def add_sample_to_grad_estimate(grad):
        # generate noise
        v = np.random.normal(scale=noise, size=num_params)

        # calculate negative/positive noise scores
        pos_rew = get_step_score(model, v)
        neg_rew = get_step_score(model, -v)

        # keep running total for gradient estimate
        if pos_rew != neg_rew:
            grad = np.where(pos_rew > neg_rew, grad + v * (pos_rew / neg_rew - 1), grad - v * (neg_rew / pos_rew - 1))

        # decay gradient est to fresh estimate over time
        grad *= momentum

        return grad

    while True:

        # base samples each iteration used to estimate gradient
        for _ in range(base_est_iters):
            grad = add_sample_to_grad_estimate(grad)

        # <<<< calculate step >>>>
        step = grad * ((noise * lr) / np.mean(np.abs(grad)))

        # clip step size to try and not exceed the noise used for gradient estimation
        step = np.clip(step, -noise * step_clip_factor, noise * step_clip_factor)

        # calculate average step magnitude for use in determining weight decay factor
        step_mag = np.mean(np.absolute(step))

        # get step score
        new_score = get_step_score(model, step)

        # take step
        set_model_params(model, get_model_params(model) + step)

        # decay weights
        set_model_params(model, get_model_params(model) * (1 - weight_decay_coeff * step_mag))

        # calculate num recalibration samples based on score drop percentage
        if new_score < score:
            drop_percent = abs(new_score / score - 1)
            recalibration_samples = int(round(drop_percent * recaliberation_factor))
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

            if lr_reduce_wait >= patience / 2:
                # allows for more patience if algo is still finding new bests at lower LRs
                patience += patience_inc
                lr /= lr_reduce_factor
                lr_reduce_wait = 0

            if iters_since_last_best >= patience:
                return score_history
