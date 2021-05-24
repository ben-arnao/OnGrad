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
        reduce_lr_patience=25,
        base_lr=1e-5,
        step_clip_factor=5,
        score_delta_warmup_scale=1,
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

    # prepare vars
    lr = base_lr
    iters_since_last_best = 0
    lr_reduce_wait = 0

    # function to calculate the score of a step (but do not actually commit the step to the model)
    def get_step_score(model, step, samples, batch_size):
        theta = get_model_params(model)
        set_model_params(model, theta + step)
        preds = model.predict(samples, batch_size=batch_size)
        score = get_score(preds)
        set_model_params(model, theta)
        return score

    def add_grad_estimate(grad):
        # generate noise
        v = np.random.normal(scale=grad_noise, size=num_params)
        min_noise = 1e-3
        while np.any(np.abs(v) < min_noise):
            v = np.where(np.abs(v) < min_noise, np.random.normal(scale=grad_noise, size=num_params), v)

        # calculate negative/positive noise scores
        pos_rew = get_step_score(model, v, train_samples, batch_size)
        neg_rew = get_step_score(model, -v, train_samples, batch_size)

        if pos_rew != 0 and neg_rew != 0:
            # add to gradient estimate
            g = (pos_rew / neg_rew - 1) / (v * 2)
            return grad * mom + g * (1 - mom)
        return grad

    while True:
        grad = add_grad_estimate(grad)

        step = grad * lr

        # clip step size to try and not exceed estimation noise
        step = np.clip(step, -grad_noise * step_clip_factor, grad_noise * step_clip_factor)

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
        if new_train_score < train_score:
            drop = abs(new_train_score / train_score - 1)
            drop_warmup = 1 + int(round(drop * score_delta_warmup_scale * (1 / (1 - mom))))
            for x in range(drop_warmup):
                grad = add_grad_estimate(grad)
                print(x, drop_warmup, np.mean(np.abs(grad)), np.max(np.abs(grad)))

        # set score
        train_score = new_train_score

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
            lr_reduce_wait = 0
        else:
            iters_since_last_best += 1
            lr_reduce_wait += 1

            if lr_reduce_wait >= reduce_lr_patience:
                lr /= 2
                lr_reduce_wait = 0

            if iters_since_last_best >= reduce_lr_patience * 2:
                return history
