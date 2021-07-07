import numpy as np
from matplotlib import pyplot


def train(

        # <><><> functions users needs to supply for their environment
        get_model_params,
        set_model_params,
        get_episode_score,  # input is model, return a positive float for score
        model,

        ### grad params ###

        noise_stddev=0.02,  # the standard deviation of noise used for estimating gradient
        # values recommendations: near 0.1-0.2 (ex. 0.005 - 0.3)

        init_est_samples=10,  # initial amount of samples used to calculating gradient for step
        # values recommendations: 5, 10, 15+ (use larger values for more complex problems)

        est_samples_reduce=1,  # amount to reduce est_samples by each LR reduce step
        # values recommendations: 1-5. Increasing this can improve sample efficiency but may harm performance

        grad_decay_factor=50,  # this can be seen as a form of momentum used in calculating gradients
        # values recommendations: 1-1000 (larger value = lower momentum, more reactive estimate)

        est_samples_floor=1,  # minimum amount of sample used to calculate gradient
        # values recommendations: keep at default

        grad_estimate_mode='additive',  # when using multiplicative, a reward increase of 10 -> 100 should be viewed
        # similarly as a change of 100 -> 1000. (multiplicative only works with positive reward)
        # values options: 'additive' or 'multiplicative'

        ### patience params ###

        patience=10,
        patience_inc=10,
        min_delta=0.01,

        lr_reduce_factor=3,  # factor to reduce average step size by (reduce LR)
        # values recommendations: keep at default

        ### step/weight params ###

        step_clip_factor=3,  # ensure that not step exceeds that standard dev of noise * X
        # values recommendations: keep at default

        init_lr=0.5,  # initial learn rate (factor)
        # values recommendations: 0.1 or 1

        weight_decay=0.1,  # weight decay *factor*. different from regular weight decay
        # values recommendations: 0.01 -> 1

        ### other ###

        init_iters=100  # if noise in weights is unable to produce a score difference after X attempts,
        # throw an error. For many problems, this should not be relevant. For problems where we expect the environment
        # to require many tries to generate varying scores, this can be increased.
        # Otherwise, see the error thrown during initialization.
):
    # get baseline
    best_score = get_episode_score(model)

    if best_score <= 0:
        # retry weight init, or manually adjust initial weights to produce actions/valid scores
        raise Exception('Model weights are not initialized to a valid starting point')

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

        if score <= 0 and grad_estimate_mode == 'multiplicative':
            raise NotImplementedError('multiplicative grad estimate mode does not accept negative score values')
        return score

    def add_sample_to_grad_estimate(grad):
        # generate noise
        v = np.random.normal(scale=noise_stddev, size=num_params)

        # calculate negative/positive noise scores
        pos_rew = get_step_score(model, v)
        neg_rew = get_step_score(model, -v)

        # keep running total for gradient estimate
        if pos_rew != neg_rew:
            if grad_estimate_mode == 'multiplicative':
                grad = np.where(pos_rew > neg_rew,
                                grad + v * (pos_rew / neg_rew - 1),
                                grad - v * (neg_rew / pos_rew - 1))
            elif grad_estimate_mode == 'additive':
                grad = np.where(pos_rew > neg_rew,
                                grad + v * (pos_rew - neg_rew),
                                grad - v * (neg_rew - pos_rew))
            else:
                raise ValueError('did not supply a valid option for \'grad_estimate_mode\'')
        else:
            print('noise in weights did not produce a change in reward. if you see too many of these messages '
                  'at the beginning of training, consider starting with bigger noise')

        return grad

    # prep variables before training
    lr = init_lr
    iters_since_last_best = 0
    lr_reduce_wait = 0
    est_samples = init_est_samples

    # ensure noise is able to produce varying scores
    i = 0
    while grad.any() == 0:
        grad = add_sample_to_grad_estimate(grad)
        i += 1
        if i >= init_iters:
            raise Exception('Supplied model is not learnable as noise in weights was not able to produce varying '
                            'scores after {0} attempts. The score difference is used to estimate the gradient. '
                            'Consider increasing the std dev of the noise, or initializing the model with weights that '
                            'produce different scores, even if occasionally. For example, one technique might be to '
                            'pre-train the model on varying samples with random actions to simulate an epsilon greedy '
                            'policy). Note: This is only required to get optimization up and running, and should be '
                            'irrelevant after optimization starts.'.format(init_iters))

    while True:

        # accumlate samples for gradient estimate
        for _ in range(est_samples):
            grad = add_sample_to_grad_estimate(grad)

        # calculate step
        step = grad * ((noise_stddev * lr) / np.mean(np.abs(grad)))

        # clip step size to try and not exceed estimation noise
        step = np.clip(step, -noise_stddev * step_clip_factor, noise_stddev * step_clip_factor)

        # calculate average step magnitude for use in determining weight decay factor
        step_mag = np.mean(np.absolute(step))

        # decay gradient based on step size
        grad_decay_perc = grad_decay_factor * step_mag
        if grad_decay_perc > 1:
            grad_decay_perc = 1
        grad *= 1 - grad_decay_perc

        # get step score
        score = get_step_score(model, step)

        # take step
        set_model_params(model, get_model_params(model) + step)

        # decay weights
        set_model_params(model, get_model_params(model) * (1 - weight_decay * step_mag))

        score_history.append(score)
        print('step:', len(score_history), 'score:', score)

        # plot score history
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
                    est_samples -= est_samples_reduce

                patience += patience_inc
                lr /= lr_reduce_factor
                lr_reduce_wait = 0

            if iters_since_last_best >= patience:
                return score_history, model

