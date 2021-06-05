import numpy as np
from matplotlib import pyplot


def train(

        # <><><> functions users needs to supply for their environment
        get_model_params,
        set_model_params,
        get_episode_score,  # input is model, return a positive float for score
        model,

        ### grad params ###
        grad_decay=0.1,
        noise_stddev=0.01,  # standard deviation of the noise used to estimate the gradient
        init_est_samples=10,
        decay_grad_every_x_samples=10,
        recaliberation_factor=25,
        est_samples_floor=1,
        extra_decay_on_drop=False,

        ### patience params ###
        patience=20,
        patience_inc=10,
        min_delta=0.01,
        lr_reduce_factor=2,

        ### step/weight params ###
        step_clip_factor=3,
        init_lr=0.1,
        weight_decay=0.1,
):
    # get baseline
    best_score = get_episode_score(model)

    # setup score history tracking
    score_history = []

    # get param count
    num_params = len(get_model_params(model))

    # gradient estimate
    grad = np.zeros(num_params, dtype=np.float32)

    # function to calculate the score of a step (but do not actually commit the step to the model)
    def get_step_score(model, step):
        theta = get_model_params(model)
        set_model_params(model, theta + step)
        score = get_episode_score(model)
        set_model_params(model, theta)

        if score <= 0:
            raise NotImplementedError('Does not currently handle negative scores or 0. consider adding a baseline and'
                                      'clipping anything still negative to 0')
        return score

    def add_sample_to_grad_estimate(grad):
        # generate noise
        v = np.random.normal(scale=noise_stddev, size=num_params)

        # calculate negative/positive noise scores
        pos_rew = get_step_score(model, v)
        neg_rew = get_step_score(model, -v)

        # keep running total for gradient estimate
        if pos_rew != neg_rew:
            grad = np.where(pos_rew > neg_rew, grad + v * (pos_rew / neg_rew - 1), grad - v * (neg_rew / pos_rew - 1))

        # eventually probably want to use score difference as opposed to percent drop so we can use negative scores
        # we will need to normalize gradient before step so that the scale of score doesn't affect step size

        return grad

    # prep variables before training
    lr = init_lr
    iters_since_last_best = 0
    lr_reduce_wait = 0
    est_samples = init_est_samples
    extra_iters = 0
    tot_est_samples = 0
    decay_grad_next_iter = False

    while True:
        if decay_grad_next_iter:
            grad *= 1 - grad_decay
            decay_grad_next_iter = False

        # accumlate samples for gradient estimate
        for _ in range(est_samples + extra_iters):
            grad = add_sample_to_grad_estimate(grad)
            tot_est_samples += 1

            # decay gradient to prevent it from stagnating when gradient is neutral
            if tot_est_samples % decay_grad_every_x_samples == 0:
                decay_grad_next_iter = True

        # calculate step
        step = grad * ((noise_stddev * lr) / np.mean(np.abs(grad)))

        # clip step size to try and not exceed estimation noise
        step = np.clip(step, -noise_stddev * step_clip_factor, noise_stddev * step_clip_factor)

        # calculate average step magnitude for use in determining weight decay factor
        step_mag = np.mean(np.absolute(step))

        # get step score
        new_score = get_step_score(model, step)

        # take step
        set_model_params(model, get_model_params(model) + step)

        # decay weights
        set_model_params(model, get_model_params(model) * (1 - weight_decay * step_mag))

        # calculate num recalibration samples based on score drop percentage
        if new_score < score and recaliberation_factor > 0:
            if extra_decay_on_drop:
                grad *= 1 - grad_decay
            drop = abs(new_score / score - 1)
            extra_iters = max(1, int(round(drop * recaliberation_factor)))
        else:
            extra_iters = 0

        # set score after calculating score delta
        score = new_score
        score_history.append(score)

        # plot train/test history
        pyplot.clf()
        pyplot.plot(score_history, linewidth=0.35)
        pyplot.savefig('score history.png')

        if score > best_score * (1 + min_delta):
            best_score = score
            iters_since_last_best = 0
            lr_reduce_wait = 0
        else:
            iters_since_last_best += 1
            lr_reduce_wait += 1

            if lr_reduce_wait >= int(patience / 2):
                if est_samples > est_samples_floor:
                    est_samples -= est_samples_floor

                patience += patience_inc
                lr /= lr_reduce_factor
                lr_reduce_wait = 0

            if iters_since_last_best >= patience:
                return score_history, model

