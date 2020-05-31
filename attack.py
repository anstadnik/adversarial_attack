import time
from multiprocessing import cpu_count
from multiprocessing.pool import Pool

import numpy as np
import pytesseract
from PIL import Image
import pickle
import atexit

from genattack_tf2 import GenAttack2

pop_size = 6
test_size = 1000
mutation_rate = 0.30
alpha = 0.5
adaptive = False
max_steps = 10000
# max_steps = 1
mutation_rate = 0.3
# eps = 0.30
eps = 1
output_dir = 'mnist_output'
temp = 0.1

rez = []

def save_rez():
    with open(f'rez_ps_{str(pop_size)}_mr_{str(mutation_rate)}.pickle', 'wb') as f:
        pickle.dump(rez, f)
    

def score(img: np.ndarray, c=6):
    img = Image.fromarray(((np.squeeze(img) + 0.5) * 255).astype(np.uint8))
    d_ = pytesseract.image_to_data(
        img, output_type=pytesseract.Output.DICT, config=f'--psm {c} --oem 0')
    return max([c for c in d_['conf'] if isinstance(c, int)] + [0])

def pred(imgs):
    with Pool(cpu_count()) as p:
        return np.array(p.map(score, imgs))
    # return np.array([score(i, c) for i in imgs])

def gen_noise(img, pop_size_ = None, mutation_rate_ = None):
    pop_size_ = pop_size_ or pop_size
    mutation_rate_ = mutation_rate_ or mutation_rate
    image_shape = img.shape
    image_channels = 3

    attack = GenAttack2(pred=pred,
                        pop_size=pop_size_,
                        mutation_rate=mutation_rate_,
                        eps=eps,
                        max_steps=max_steps,
                        alpha=alpha,
                        image_shape=image_shape,
                        image_channels=image_channels,
                        temp=temp,
                        adaptive=adaptive)

    start_time = time.time()
    result = attack.attack(img)
    end_time = time.time()
    attack_time = (end_time-start_time)
    if result is not None:
        adv_img, query_count, margin_log = result
        rez.append(margin_log)
        # if len(rez) >= 200:
        #     print('bye')
        #     exit()
        # final_pred = sess.run(test_pred, feed_dict={
        #                       test_in: [adv_img]})[0]
    else:
        print('Attack failed')
    return result

import atexit
atexit.register(save_rez)
