import numpy as np
from matplotlib import pyplot
import copy


def train(

        # <><><> functions users needs to supply for their environment
        get_model_params,
        set_model_params,
        get_episode_score,  # input is model, return a positive float for score
        model,
        init_routine,  # custom routine used to initialize weights to good starting point

        ### grad estimate params ###
        momentum=0.99,
        grad_est_bounds_factor=1,
        noise_stddev=0.02,

        ### patience/reduce params ###
        patience=20,
        min_delta=0.01,
        lr_reduce_factor=2,
        noise_reduce_factor=1,

        ### step/weight params ###
        step_clip_factor=1,  # ensure that not step exceeds that standard dev of noise * X
        # values recommendations: keep at default

        init_lr=1,  # initial learn rate (factor)
        # values recommendations: 0.1 or 1

        weight_decay=0.1,  # weight decay *factor*. different from regular weight decay
        # values recommendations: 0.01 -> 1

        ### other ###
        init_iters=100  # if noise in weights is unable to produce a score difference after X attempts,
        # throw an error. For many problems, this should not be relevant. For problems where we expect the environment
        # to require many tries to generate varying scores, this can be increased.
        # Otherwise, see the error thrown during initialization.
):
    
    init_routine(model)
    
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
        return score

    def add_sample_to_grad_estimate(grad):
        # generate noise
        v = np.random.normal(scale=noise_stddev, size=num_params)

        # calculate negative/positive noise scores
        pos_rew = get_step_score(model, v)
        neg_rew = get_step_score(model, -v)

        if pos_rew != neg_rew:

            m = momentum

            if pos_rew > neg_rew:
                grad = grad * m + np.sign(v) * (1 - m)
            else:
                grad = grad * m - np.sign(v) * (1 - m)

            return grad
        else:
            return None

    # prep variables before training
    lr = init_lr
    iters_since_last_best = 0
    lr_reduce_wait = 0

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

        # estimate gradient for step
        grad_hi = copy.deepcopy(grad)
        grad_lo = copy.deepcopy(grad)

        is_new_high = np.ones(grad.shape, dtype=bool)
        is_new_low = np.ones(grad.shape, dtype=bool)

        # here we keep accumulating samples and adding to estimate until 'grad_est_threshold' percent of gradient
        # estimate stays within 'grad_min_delta_perc' of the current bounds
        
        consec_no_change = 0

        # in other words, we keep estimating gradient until the estimate is stationary enough
        while sum(np.logical_or(is_new_high, is_new_low)) != 0:

            new_grad = add_sample_to_grad_estimate(grad)
            if new_grad is not None:
                grad = new_grad
            else:
                consec_no_change += 1

                if consec_no_change > 25:
                    print('estimation surface too flat (noise too small, or not able to produce varying scores)')
                    return score_history
                continue

            is_new_high = np.where(grad > grad_hi + grad_est_bounds_factor * (1 - momentum), True, False)
            grad_hi = np.where(is_new_high, grad, grad_hi)

            is_new_low = np.where(grad < grad_lo - grad_est_bounds_factor * (1 - momentum), True, False)
            grad_lo = np.where(is_new_low, grad, grad_lo)

        # calculate step
        step = grad * ((noise_stddev * lr) / np.mean(np.abs(grad)))

        # clip step size to try and not exceed estimation noise
        step = np.clip(step, -noise_stddev * step_clip_factor, noise_stddev * step_clip_factor)

        # calculate average step magnitude for use in determining weight decay factor
        step_mag = np.mean(np.absolute(step))

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
                noise_stddev /= noise_reduce_factor
                lr /= lr_reduce_factor
                lr_reduce_wait = 0

            if iters_since_last_best >= patience:
                return score_history, model
