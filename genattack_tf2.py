import time
import random
import numpy as np
from PIL import Image
from scipy.special import softmax
from sklearn.preprocessing import OneHotEncoder


def imresize(img, size):
    resized_img = Image.fromarray(img).resize(size=size)
    resized_img = np.array(resized_img)
    return resized_img


class GenAttack2(object):
    def mutation_op(self,  cur_pop, idx, step_noise=0.01, p=0.005):
        perturb_noise = np.random.uniform(low=-step_noise, high=step_noise,
                                          size=cur_pop.shape)
        mutated_pop = perturb_noise * \
            (np.random.uniform(size=cur_pop.shape) < p).astype(np.float32) + cur_pop
        return mutated_pop

    def attack_step(self, idx, success, orig_copies, cur_noise, prev_elite, margin_log, best_win_margin, cur_plateau_count, num_plateaus):
        noise_shape = self.image_shape
        cur_pop = np.clip(
            cur_noise + orig_copies, self.box_min, self.box_max)
        # List of predictions of shape (n, 10)
        pop_preds = self.pred(cur_pop)

        # Test if there is a succesfull result
        success_pop = np.equal(pop_preds, 0).astype(np.int32)
        success = np.max(success_pop, axis=0)
        # if success:
        #     __import__('ipdb').set_trace()

        # the goal is to maximize this loss
        loss = -np.log(pop_preds+1e-30)

        # Difference between prediction of the target label and the maximum one
        win_margin = np.min(pop_preds, axis=0)

        # If prob of target label became larger
        if np.less(win_margin, best_win_margin):
            new_best_win_margin, new_cur_plateau_count = win_margin, 0
        else:
            new_best_win_margin, new_cur_plateau_count = best_win_margin, cur_plateau_count+1,

        # If probability of target is very small, bigger threshold
        if np.greater(win_margin, -0.40):
            plateau_threshold = 100
        else:
            plateau_threshold = 300

        # If plateau lasts for long enough
        if np.greater(new_cur_plateau_count, plateau_threshold):
            new_num_plateaus, new_cur_plateau_count = num_plateaus+1, 0,
        else:
            new_num_plateaus, new_cur_plateau_count = num_plateaus, new_cur_plateau_count

        if self.adaptive:
            step_noise = np.maximum(self.alpha,
                                    0.4*np.power(0.9,
                                                 new_num_plateaus.astype(np.float32)))
            if np.less(idx, 10):
                step_p = 1.0
            else:
                step_p = np.maximum(self.mutation_rate,
                                    0.5*np.power(0.90, new_num_plateaus.astype(np.float32)))
        else:
            step_noise = self.alpha
            step_p = self.mutation_rate

        step_temp = self.temp

        # If found a solution
        if np.equal(success, 1):
            # Preserve it
            elite_idx = np.expand_dims(
                np.argmax(success_pop).astype(np.int32), axis=0)
        else:
            # Save the best entity
            elite_idx = np.expand_dims(
                np.argmax(loss, axis=0).astype(np.int32), axis=0)
        elite = cur_noise[elite_idx]

        # Get fitness for each value. If step_temp is smaller, then the function
        # will be steeper
        select_probs = softmax(np.squeeze(loss) / step_temp)

        # Might output a single item repeated
        parents = np.random.choice(
            len(select_probs), p=select_probs, size=2*self.pop_size-2)

        # First half of parents
        parent1 = cur_noise[parents[:self.pop_size-1]]
        # Second half of parents
        parent2 = cur_noise[parents[self.pop_size-1:]]
        # Parent probabilities
        pp1 = select_probs[parents[:self.pop_size-1]]
        pp2 = select_probs[parents[self.pop_size-1:]]
        pp2 = pp2 / (pp1+pp2)
        # Expand shape of probabilities to (n, 1, 1, 1), and then repeat them so
        # they will match shape of noise
        pp2 = np.tile(np.expand_dims(pp2, (1, 2, 3)),
            (1, noise_shape[0], noise_shape[1], self.image_channels))
        # Boolean mask which determines which parents' genes should be taken
        xover_prop = (np.random.uniform(
            size=parent1.shape) > pp2).astype(np.float32)
        childs = parent1 * xover_prop + parent2 * (1-xover_prop)
        # Index of iteration
        idx += 1
        # print(idx, np.max(loss), win_margin, step_p, step_noise,
        #         new_cur_plateau_count, pop_preds)
        margin_log = np.concatenate([margin_log, [[win_margin]]], axis=0)
        # Apply mutation to the population
        mutated_childs = self.mutation_op(
            childs, idx=idx, step_noise=self.eps*step_noise, p=step_p)
        # Add elite to the population
        new_pop = np.concatenate((mutated_childs, elite), axis=0)
        return idx, success, orig_copies, new_pop, np.reshape(elite,
                (noise_shape[0], noise_shape[1], self.image_channels)), margin_log, new_best_win_margin, new_cur_plateau_count, new_num_plateaus

    def __init__(self, pred, pop_size=6, mutation_rate=0.001,
                 eps=0.15, max_steps=10000, alpha=0.20,
                 image_shape=299,
                 image_channels=3,
                 temp=0.3,
                 adaptive=False):
        self.eps = eps
        self.pop_size = pop_size
        self.pred = pred
        self.alpha = alpha
        self.temp = temp
        self.max_steps = max_steps
        self.mutation_rate = mutation_rate
        self.image_shape = image_shape
        noise_shape = self.image_shape
        self.image_channels = image_channels
        self.adaptive = adaptive
        self.input_img = np.zeros(
            (1, self.image_shape[0], self.image_shape[1], self.image_channels), dtype=np.float32)
        # copies of original image
        self.pop_orig = np.zeros(
            (self.pop_size, self.image_shape[0], self.image_shape[1], image_channels), dtype=np.float32)
        self.pop_noise = np.zeros(
            (self.pop_size, noise_shape[0], noise_shape[1], self.image_channels), dtype=np.float32)

        self.init_success = 0
        self.box_min = np.tile(np.maximum(
            self.input_img-eps, -0.5), (self.pop_size, 1, 1, 1))
        self.box_max = np.tile(np.minimum(
            self.input_img+eps, 0.5), (self.pop_size, 1, 1, 1))
        self.margin_log = np.zeros((1, 1), dtype=np.float32)
        self.i = 0

        # Variables to detect plateau
        self.best_win_margin = -1
        self.cur_plateau_count = 0
        self.num_plateaus = 0

    def attack_main(self):
        while np.logical_and(np.less_equal(self.i, self.max_steps),
                             np.equal(self.init_success, 0)):
            (self.i, self.init_success, self.pop_orig,
             self.pop_noise, self.pop_noise[0], self.margin_log,
             self.best_win_margin, self.cur_plateau_count,
             self.num_plateaus) = self.attack_step(self.i, self.init_success, self.pop_orig,
                                                   self.pop_noise, self.pop_noise[0], self.margin_log,
                                                   self.best_win_margin, self.cur_plateau_count,
                                                   self.num_plateaus)
        return (self.i, self.init_success,  self.pop_orig, self.pop_noise,
                self.pop_noise[0], self.margin_log, self.best_win_margin,
                self.cur_plateau_count, self.num_plateaus)

    def initialize(self, img):
        # Add dimension to the input_img (from (28, 28, 1) to (1, 28, 28, 1))
        self.input_img = np.expand_dims(img, axis=0)
        # Stack input images (from (1, 28, 28, 1) to (n, 28, 28, 1))
        self.pop_orig = np.tile(self.input_img, [self.pop_size, 1, 1, 1])
        # Init initial population.
        self.pop_noise = self.mutation_op(
            self.pop_noise, idx=self.i, p=self.mutation_rate, step_noise=self.eps)
        self.margin_log = np.zeros((1, 1), dtype=np.float32)
        self.best_win_margin = np.array(-1.0, dtype=np.float32)
        self.cur_plateau_count = np.array(0, dtype=np.int32)
        self.num_plateaus = 0
        # print('Population initailized')

    def attack(self, input_img):
        self.initialize(input_img)
        (num_steps, success,  copies, final_pop, adv_noise,
         log_hist, _, _, _) = self.attack_main()
        if success:
            adv_img = np.clip(np.expand_dims(input_img, axis=0)+np.expand_dims(adv_noise, axis=0),
                              self.box_min[0:1], self.box_max[0:1])

            # Number of queries = NUM_STEPS * (POP_SIZE -1 ) + 1
            # We subtract 1 from pop_size, because we use elite mechanism, so one population
            # member is copied from previous generation and no need to re-evaluate it.
            # The first population is an exception, therefore we add 1 to have total sum.
            query_count = num_steps * (self.pop_size - 1) + 1
            return adv_img[0], query_count, log_hist[1:, :]
        else:
            return None
