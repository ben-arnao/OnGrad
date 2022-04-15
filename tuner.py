import numpy as np
from matplotlib import pyplot
import copy


def train(

        # <><><> functions users needs to supply for their environment
        get_model_params,
        set_model_params,
        get_episode_score,  # input is model, return's float for score
        model,
        init_routine,  # custom routine used to initialize weights to good starting point

        ### grad estimate params ###
        momentum=0.99,  # determines the stability/accuracy of the gradient estimate. recommended 0.99 - 0.999+
        # sometimes a lower value can act as a form of regularization
        est_threshold=0.9,  # determines how lenient to be with the quality of a step.
        # a value too high will cause us to improve the quality of the gradient beyond what actually has an
        # impact on performance, and therefore result in poor sample efficiency. recommended 0.8 - 0.95+
        init_noise_stddev=0.1,  # this value is good in most cases.
        # can potentially be put higher to increase speed/broadness of initial search area,
        # or lower if optimization fails to get off the ground (more sensitive models)

        ### patience/reduce params ###
        patience=10,  # patience used to derive early stopping and noise reduce patience. default value recommended
        noise_reduce_factor=2,  # factor to reduce noise by when score plateaus. recommended either 2 (safer option) or
        # 10 (faster option)

        ### other ###
        init_iters=100,  # if noise in weights is unable to produce a score difference after X attempts,
        # throw an error. For many problems, this should not be relevant. For problems where we expect the environment
        # to require many tries to generate varying scores, this can be increased.
        # otherwise, see the error thrown during initialization.

        consec_no_change_thresh=25,  # if noise is unable to produce varying scores for X iters, training is terminated.
        # it may be normal for *some* iterations to not produce different scores
):
    print('performing initialization routine...')
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

        # noise did not produce a change in score
        if pos_rew == neg_rew:
            return None

        if pos_rew > neg_rew:
            return grad * momentum + np.sign(v) * (1 - momentum)
        else:
            return grad * momentum - np.sign(v) * (1 - momentum)

    # prep variables before training
    noise_stddev = init_noise_stddev
    iters_since_last_best = 0
    noise_reduce_wait = 0
    grad_hi = copy.deepcopy(grad)
    grad_lo = copy.deepcopy(grad)

    # ensure noise is able to produce varying scores
    i = 0
    while grad.any() == 0:
        new_grad = add_sample_to_grad_estimate(grad)
        if new_grad is not None:
            grad = new_grad
        i += 1
        if i >= init_iters:
            raise Exception('Supplied model is not learnable as noise in weights was not able to produce varying '
                            'scores after {0} attempts. The score difference is used to estimate the gradient. '
                            'Consider increasing the std dev of the noise, or initializing the model with weights that '
                            'produce different scores, even if occasionally. For example, one technique might be to '
                            'pre-train the model on varying samples with random actions to simulate an epsilon greedy '
                            'policy). Note: This is only required to get optimization up and running, and should be '
                            'irrelevant after optimization starts.'.format(init_iters))

    print('--- model is able to produce varying scores. training started! ---')
    while True:  # estimate gradient for a single step

        # calculate new initial bounds based on distance previous estimate traveled
        bound_mag = np.where(np.abs(grad_hi - grad) > np.abs(grad - grad_lo),
                             np.abs(grad - grad_lo),
                             np.abs(grad_hi - grad))
        grad_hi = grad + bound_mag
        grad_lo = grad - bound_mag

        is_new_high = np.ones(grad.shape, dtype=bool)
        is_new_low = np.ones(grad.shape, dtype=bool)
        consec_no_change = 0

        # here we keep accumulating samples and adding to estimate until the percent of stationary estimates vs.
        # non-stationary estimates goes above the user defined threshold

        while sum(np.logical_or(is_new_high, is_new_low)) / num_params > 1 - est_threshold:

            new_grad = add_sample_to_grad_estimate(grad)
            if new_grad is not None:
                grad = new_grad
                consec_no_change = 0
            else:
                consec_no_change += 1

                if consec_no_change > consec_no_change_thresh:
                    print('\tnoise not big enough to produce different scores. increasing noise...', noise_stddev)
                    noise_stddev *= noise_reduce_factor
                    noise_reduce_wait = 0
                    consec_no_change = 0
                continue

            is_new_high = np.where(grad > grad_hi, True, False)
            grad_hi = np.where(is_new_high, grad, grad_hi)

            is_new_low = np.where(grad < grad_lo, True, False)
            grad_lo = np.where(is_new_low, grad, grad_lo)

        # calculate step
        step = grad * noise_stddev

        # take step
        set_model_params(model, get_model_params(model) + step)

        # get step score
        score = get_episode_score(model)

        score_history.append(score)
        print('step #{} | score: {}'.format(len(score_history), score))

        # plot score history
        pyplot.clf()
        pyplot.plot(score_history, linewidth=0.35)
        pyplot.savefig('score history.png')

        if score > best_score:
            best_score = score
            iters_since_last_best = 0
            noise_reduce_wait = 0
        else:
            iters_since_last_best += 1
            noise_reduce_wait += 1

            if noise_reduce_wait >= patience:
                noise_stddev /= noise_reduce_factor
                noise_reduce_wait = 0
                print('\treducing noise...', noise_stddev)

            if iters_since_last_best >= patience * 2:
                print('training complete!')
                return
