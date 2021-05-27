import numpy as np
from matplotlib import pyplot


def train(
        
        # <><><> functions users needs to supply for their environment
        
        get_score,  # accepts an array of predictions, returns a float value representing score
        get_model_params,
        set_model_params,
        model_predict,  # user defines function where input is (model, samples) and output is predictions
        
        model,  # supplied model, can be a pytorch or tensorflow model
        
        # supply samples
        train_samples,
        test_samples=None,  # users doesn't need to supply test samples
        
        # <><><> training params

        # reduce lr and end training patience
        lr_patience=50,
        end_patience=1000,

        grad_noise=2e-2,  # noise in weights used to estimate gradient. this is very problem dependant
        # but the default should be good enough in most cases
        
        weight_decay_coeff=0.1,  # weight decay = mean absolute step size * 'weight_decay_coeff'
        
        lr=1e-1,  # different than standard LR since grad is calculated differently. start out high
        # default value probably fine in most cases
        
        step_clip_factor=5,  # we do not want to take steps that exceed the size of noise used for grad estimate
        # recommended 3-10
        
        score_delta_warmup_scale=100,  # that amount of estimate recalibration iters taken after score drops
        # use a higher value if you see that your model is not recovering well after drops
        
        grad_decay=0.1,  # higher value == more adaptive estimate, lower value == more stable and slow moving
        
        ):

    # get baselines
    train_preds = model_predict(model, train_samples)
    best_train_score = get_score(train_preds)
    test_preds = model_predict(model, train_samples)
    best_test_score = get_score(test_preds)

    # setup score history tracking
    train_score = best_train_score
    history = {'train': [best_train_score], 'test': [best_test_score]}

    # get param count
    num_params = len(get_model_params(model))

    # gradient estimate
    grad = np.zeros(num_params, dtype=np.float32)

    # prepare vars
    iters_since_last_best = 0
    lr_reduce_wait = 0

    # function to calculate the score of a step (but do not actually commit the step to the model)
    def get_step_score(model, step, samples):
        theta = get_model_params(model)
        set_model_params(model, theta + step)
        preds = model_predict(model, samples)
        score = get_score(preds)
        set_model_params(model, theta)
        return score

    def add_sample_to_grad_estimate(grad):
        # generate noise
        v = np.random.normal(scale=grad_noise, size=num_params)

        # calculate negative/positive noise scores
        pos_rew = get_step_score(model, v, train_samples)
        neg_rew = get_step_score(model, -v, train_samples)

        # keep running total for gradient estimate
        if pos_rew != neg_rew and pos_rew != 0 and neg_rew != 0:
            score_difference = abs(pos_rew - neg_rew)
            grad = np.where(pos_rew > neg_rew, grad + v * score_difference, grad - v * score_difference)

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

        # get step score for train and test
        new_train_score = get_step_score(model, step, train_samples)
        if train_samples is not None:
            test_score = get_step_score(model, step, test_samples)
            history['test'].append(test_score)

        # take step
        set_model_params(model, get_model_params(model) + step)

        # decay weights where weight decay value is modified by the step magnitude
        # (this is why weight decay should be a higher value by default)
        set_model_params(model, get_model_params(model) * (1 - weight_decay_coeff * step_mag))

        # If score drops, calculate additional samples for gradient estimate.
        # The larger the drop, the more samples will be calculated and the more
        # accurate and "up to date" the estimate will be.
        if new_train_score < train_score:
            drop_percent = abs(new_train_score / train_score - 1)

            # calculate num samples based on drop percentage, momentum (higher mom, more samples), and
            # a user-supplied coefficient
            recalibration_samples = 1 + int(round(drop_percent * score_delta_warmup_scale))
            for _ in range(recalibration_samples):
                grad = add_sample_to_grad_estimate(grad)

        # set score after calculating score delta
        train_score = new_train_score
        history['train'].append(train_score)

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

            # reduce lr/grad decay
            if lr_reduce_wait >= lr_patience:
                
                # since we take less recalibration steps and lr gets lower it will take longer for estimate to saturate
                lr_patience *= 2
                lr /= np.sqrt(10)
                lr_reduce_wait = 0

            if iters_since_last_best >= end_patience:
                return history
