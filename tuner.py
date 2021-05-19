from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
import numpy as np
from matplotlib import pyplot
from tensorflow_addons.activations import mish


def train(
        get_score,
        get_model_params,
        set_model_params,
        train_samples,
        test_samples,
        output_size,
        mom=0.99,
        grad_noise=1e-2,
        weight_decay=0.1,
        stop_patience=100,
        adaptive_lr_scale=0.001,
        base_lr=1e-5,
        score_delta_lr_scale=0.1,
        grad_delta=0.1,
        n_layers=4,
        kernel_size=200,
        activation='mish',
        batch_size=32768):

    # make model
    model_arch = [kernel_size for _ in range(n_layers)]
    model = Sequential()
    for k in model_arch:
        if kernel_size > 0:
            if activation == 'mish':
                model.add(Dense(k))
                model.add(Activation(mish))
            else:
                model.add(Dense(k, activation=activation))
    model.add(Dense(output_size))
    model.compile()

    # get baselines
    train_preds = model.predict(train_samples, batch_size=batch_size)
    best_train_score = get_score(train_preds)
    test_preds = model.predict(test_samples, batch_size=batch_size)
    best_test_score = get_score(test_preds)

    # setup score history tracking
    train_score = best_train_score
    history = {'train': [best_train_score], 'test': [best_test_score]}

    # get param count
    num_params = len(get_model_params(model))

    # grad tracking vars
    grad = np.zeros(num_params, dtype=np.float32)
    grad_est_anchor = np.zeros(num_params, dtype=np.float32)

    # prepare vars
    lr = base_lr
    adaptive_lr = 1
    iters_since_last_best = 0
    iters = 0

    # the recommended amount of samples to satisfy the rule of thumb that for a momentum value of X,
    # 3 / (1 - X) is the amount of samples needed to get an accurate estimation
    warmup = int(round(3 / (1 - mom)))

    # function to calculate the score of a step (but do not actually commit the step to the model)
    def get_step_score(model, step, samples, batch_size):
        theta = get_model_params(model)
        set_model_params(model, theta + step)
        preds = model.predict(samples, batch_size=batch_size)
        score = get_score(preds)
        set_model_params(model, theta)
        return score

    while True:
        # generate noise
        v = np.random.normal(scale=grad_noise, size=num_params)

        # we do not want noise too small becuase this causes a misleadingly large calculation for gradient
        min_noise = grad_noise / 10
        while np.any(np.abs(v) < min_noise):
            v = np.where(np.abs(v) < min_noise, np.random.normal(scale=grad_noise, size=num_params), v)

        # calculate noise score
        pos_rew = get_step_score(model, v, train_samples, batch_size)
        neg_rew = get_step_score(model, -v, train_samples, batch_size)

        # add to gradient estimate
        g = (pos_rew - neg_rew) / (v * 2)
        grad = grad * mom + g * (1 - mom)

        # check if grad is outside current bounds
        is_new_high = np.where(grad > grad_est_anchor + grad_delta, True, False)
        is_new_low = np.where(grad < grad_est_anchor - grad_delta, True, False)
        mom_change = np.logical_or(is_new_high, is_new_low)

        # update anchor
        grad_est_anchor = np.where(mom_change, grad, grad_est_anchor)

        # get percent of grad ests that are non-stationary
        est_chg_perc = np.sum(mom_change) / len(grad)

        # if more estimates are stationary than non-stationary, start increasing lr
        # if more estimates are non-stationary than stationary, start decreasing lr
        # scale so that values near 0% and 100% cause a bigger change
        if est_chg_perc < 0.5:
            adaptive_lr *= 1 + ((0.5 - est_chg_perc) * 2 * adaptive_lr_scale)
        else:
            adaptive_lr *= 1 / (1 + (est_chg_perc - 0.5) * 2 * adaptive_lr_scale)

        # calculate warmup coeff based on current number of iters
        # scales exponentially so that in the beginning stages, we are mostly just
        # ramping our gradient estimation up before actually taking any steps
        warmup_coefficient = np.power((iters + 1) / warmup, 2) if iters < warmup else 1

        # calculate step with score delta and grad stationality modifiers
        step_size = lr * adaptive_lr * warmup_coefficient
        step = grad * step_size

        # calculate average step magnitude
        step_mag = np.mean(np.absolute(step))

        # get step score for train and test
        new_train_score = get_step_score(model, step, train_samples, batch_size)
        test_score = get_step_score(model, step, test_samples, batch_size)

        # take step
        set_model_params(model, get_model_params(model) + step)

        # decay weights where weight decay value is modified by the step magnitude
        # (this is why weight decay should be a higher value by default)
        set_model_params(model, get_model_params(model) * (1 - weight_decay * step_mag))

        # calculate score delta
        score_delta = new_train_score / train_score - 1
        # now that we calculated score delta, assign it back to original var
        train_score = new_train_score

        # if score is changing, apply modifier to lr (score drops = smaller step, score increases = bigger step)
        if score_delta != 0:
            score_delta_scaled = abs(score_delta) * score_delta_lr_scale

            # calculate score delta modifier
            if lr < base_lr and score_delta > 0:
                lr *= 1 + score_delta_scaled
                if lr > base_lr:
                    lr = base_lr
            else:
                lr *= 1 / (1 + score_delta_scaled)

        # track optimization
        history['train'].append(train_score)
        history['test'].append(test_score)

        # plot train/test history
        pyplot.clf()
        pyplot.subplot(211)
        pyplot.title('train score')
        pyplot.plot(history['train'], linewidth=0.35)
        pyplot.subplot(212)
        pyplot.title('test score')
        pyplot.plot(history['test'], linewidth=0.35)
        pyplot.savefig('score history.png')

        if train_score > best_train_score:
            best_train_score = train_score
            iters_since_last_best = 0
        else:
            iters_since_last_best += 1

            if iters_since_last_best >= stop_patience:
                return history['train'][-1]
