"""
Author: Moustafa Alzantot (malzantot@ucla.edu)

"""
import time
import os
import sys
import random
import numpy as np
import pickle

from mnist.mnist import pred, generate_data, _MNISTModel
from genattack_tf2 import GenAttack2
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
from tqdm import tqdm

model = 'mnist'
test_size = 1000
mutation_rate = 0.30
alpha = 0.5
adaptive = False
max_steps = 10000
# max_steps = 1
eps = 0.30
output_dir = 'mnist_output'
pop_size = 4
temp = 0.1

def test(pop_size, mutation_rate):
    model = _MNISTModel('mnist/models/mnist', use_log=True)
    image_dim = 28
    image_channels = 1
    num_labels = 10
    inputs, targets, reals = generate_data(test_size)

    attack = GenAttack2(pred=pred,
                        pop_size=pop_size,
                        mutation_rate=mutation_rate,
                        eps=eps,
                        max_steps=max_steps,
                        alpha=alpha,
                        image_dim=image_dim,
                        image_channels=image_channels,
                        num_labels=num_labels,
                        temp=temp,
                        adaptive=adaptive)
    num_valid_images = len(inputs)
    total_count = 0  # Total number of images attempted
    success_count = 0
    rez = []
    for ii in range(num_valid_images):
        input_img = inputs[ii]
        target_label = np.argmax(targets[ii])
        real_label = reals[ii]
        orig_pred = np.argmax(pred([input_img]))
        # orig_pred = sess.run(test_pred, feed_dict={
        #                      test_in: [input_img]})[0]
        print('Real = {}, Predicted = {}, Target = {}'.format(
            real_label, orig_pred, target_label))
        # __import__('ipdb').set_trace()
        if orig_pred != real_label:
            print('\t Skipping incorrectly classified image.')
            continue
        total_count += 1
        start_time = time.time()
        result = attack.attack(input_img, target_label)
        end_time = time.time()
        attack_time = (end_time-start_time)
        if result is not None:
            rez.append(result[2])
            adv_img, query_count, margin_log = result
            final_pred = np.argmax(pred([input_img]))
            # final_pred = sess.run(test_pred, feed_dict={
            #                       test_in: [adv_img]})[0]
            if (final_pred == target_label):
                success_count += 1
                print('--- SUCCEEEED ----')
                if image_channels == 1:
                    input_img = input_img[:, :, 0]
                    adv_img = adv_img[:, :, 0]
        else:
            rez.append(None)
            print('Attack failed')
    print('Number of success = {} / {}.'.format(success_count, total_count))
    return rez

def test_kwargs(kwargs):
    return test(**kwargs)

if __name__ == '__main__':
    params = []
    for pop_size in [2, 4, 10, 50, 100]:
        for mutation_rate in [0.3, 0.1, 0.05, 0.01]:
            params.append({'pop_size': pop_size,
                'mutation_rate': mutation_rate})

    with Pool(cpu_count(), initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
        rez = p.map(test_kwargs, params)
    rez = [{**{k: v for k, v in p.items()}, 'margin_log': m}
           for p, m in zip(params, rez)]

    with open('rezults_stolen.pickle', 'wb') as f:
        pickle.dump(rez, f)
