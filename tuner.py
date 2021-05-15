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
        weight_decay=0.01,
        stop_patience=100,
        score_delta_lr_inc_ratio=0.01,
        adaptive_lr_scale=0.1,
        lr=1e-6,
        score_delta_lr_scale=0.01,
        grad_delta=1e-4,
        grad_bounds_decay=0.001,
        n_layers=4,
        kernel_size=300,
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

    train_score = best_train_score
    history = {'train': [best_train_score], 'test': [best_test_score]}

    # get param count
    num_params = len(get_model_params(model))

    # grad tracking vars
    grad = np.zeros(num_params, dtype=np.float32)
    hi = np.zeros(num_params, dtype=np.float32)
    lo = np.zeros(num_params, dtype=np.float32)
    adaptive_lr = np.ones(num_params, dtype=np.float32)

    iters_since_last_best = 0

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
        is_new_high = np.where(grad > hi + grad_delta, True, False)
        is_new_low = np.where(grad < lo - grad_delta, True, False)
        mom_change = np.logical_or(is_new_high, is_new_low)

        # update upper and lower grad bounds
        hi = np.where(grad > hi + grad_delta, grad, hi)
        lo = np.where(grad < lo - grad_delta, grad, lo)

        # decay bounds
        hi *= 1 - grad_bounds_decay
        lo *= 1 - grad_bounds_decay

        # adjust lr based on how stationary the moving average estimate is
        adaptive_lr = np.where(mom_change,
                               adaptive_lr / (1 + adaptive_lr_scale),
                               adaptive_lr * (1 + adaptive_lr_scale))

        # transform adaptive LR to adaptive component (see notes)
        adaptive_component = np.where(adaptive_lr > 1, np.sqrt(adaptive_lr), np.power(adaptive_lr, 2))

        # calculate final step step
        step = grad * lr * adaptive_component

        # calculate average step magnitude
        step_mag = np.mean(np.absolute(step))

        # get step score
        new_train_score = get_step_score(model, step, train_samples, batch_size)

        # get step score
        test_score, test_num_trades, test_bline = get_step_score(model, step, test_samples, batch_size)

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
            if score_delta > 0:
                # in most cases we care far more about dropping the LR when the score drops than increasing step size
                # when score increases. this is to combat the "falling off a cliff" issue with reinforcement learning
                # where an agent may fall off a cliff and not be able to recover back to where it was before.
                # while it may speed up convergeance and provide other benefits to gradually ramp up LR
                # when score is increasing, this coefficient allows us have big drops in step size when the score drops
                # a big amount, but not have to worry about increasing the step size too much as time goes on and our
                # score gets higher.
                score_delta_scaled *= score_delta_lr_inc_ratio
                f = (1 + score_delta_scaled)
            else:
                f = 1 / (1 + score_delta_scaled)

            # apply modifier to lr
            lr *= f

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
