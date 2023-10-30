import math
import random
import torch
import pandas as pd
import numpy as np

np.seterr(divide='ignore', invalid='ignore')
from sklearn.metrics import *


# Defining labels for the edges
NEG_HIST = 0
NEG_RND = 1
POS_HIST = 2
POS_INDUC = 3

### Utility function and class
class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        self.epoch_count += 1

        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        return self.num_round >= self.max_round


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list, seed=None):
        self.seed = None
        self.neg_sample = 'rnd'  # negative edge sampling method: random
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def sample(self, size):
        if self.seed is None:
            src_index = np.random.randint(0, len(self.src_list), size)
            dst_index = np.random.randint(0, len(self.dst_list), size)
        else:
            src_index = self.random_state.randint(0, len(self.src_list), size)
            dst_index = self.random_state.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


# class RandEdgeSampler_NRE(object):
#     """
#     ~ "history"
#     Random Negative Edge Sampling: NRE: "Non-Repeating Edges" are randomly sampled to make task more complicated
#     Note: the edge history is constructed in a way that it inherently preserve the direction information
#     Note: we consider that randomly sampled edges come from two sources:
#       1. some are randomly sampled from all possible pairs of edges
#       2. some are randomly sampled from edges seen before but are not repeating in current batch
#     """
#
#     def __init__(self, src_list, dst_list, ts_list, seed=None, rnd_sample_ratio=0):
#         """
#     'src_list', 'dst_list', 'ts_list' are related to the full data! All possible edges in train, validation, test
#     """
#         self.seed = None
#         self.neg_sample = 'nre'  # negative edge sampling method: non-repeating edges
#         self.rnd_sample_ratio = rnd_sample_ratio
#         self.src_list = src_list
#         self.dst_list = dst_list
#         self.ts_list = ts_list
#         self.src_list_distinct = np.unique(src_list)
#         self.dst_list_distinct = np.unique(dst_list)
#         self.ts_list_distinct = np.unique(ts_list)
#         self.ts_init = min(self.ts_list_distinct)
#
#         if seed is not None:
#             self.seed = seed
#             np.random.seed(self.seed)
#             self.random_state = np.random.RandomState(self.seed)
#
#     def get_edges_in_time_interval(self, start_ts, end_ts):
#         """
#     return edges of a specific time interval
#     """
#         valid_ts_interval = (self.ts_list >= start_ts) * (self.ts_list <= end_ts)
#         interval_src_l = self.src_list[valid_ts_interval]
#         interval_dst_l = self.dst_list[valid_ts_interval]
#         interval_edges = {}
#         for src, dst in zip(interval_src_l, interval_dst_l):
#             if (src, dst) not in interval_edges:
#                 interval_edges[(src, dst)] = 1
#         return interval_edges
#
#     def get_difference_edge_list(self, first_e_set, second_e_set):
#         """
#     return edges in the first_e_set that are not in the second_e_set
#     """
#         difference_e_set = set(first_e_set) - set(second_e_set)
#         src_l, dst_l = [], []
#         for e in difference_e_set:
#             src_l.append(e[0])
#             dst_l.append(e[1])
#         return np.array(src_l), np.array(dst_l)
#
#     def sample(self, size, current_split_start_ts, current_split_end_ts):
#         history_e_dict = self.get_edges_in_time_interval(self.ts_init, current_split_start_ts)
#         current_split_e_dict = self.get_edges_in_time_interval(current_split_start_ts, current_split_end_ts)
#         non_repeating_e_src_l, non_repeating_e_dst_l = self.get_difference_edge_list(history_e_dict,
#                                                                                      current_split_e_dict)
#
#         num_smp_rnd = int(self.rnd_sample_ratio * size)
#         num_smp_from_hist = size - num_smp_rnd
#         if num_smp_from_hist > len(non_repeating_e_src_l):
#             num_smp_from_hist = len(non_repeating_e_src_l)
#             num_smp_rnd = size - num_smp_from_hist
#
#         replace = len(self.src_list_distinct) < num_smp_rnd
#         rnd_src_index = np.random.choice(len(self.src_list_distinct), size=num_smp_rnd, replace=replace)
#
#         replace = len(self.dst_list_distinct) < num_smp_rnd
#         rnd_dst_index = np.random.choice(len(self.dst_list_distinct), size=num_smp_rnd, replace=replace)
#
#         replace = len(non_repeating_e_src_l) < num_smp_from_hist
#         nre_e_index = np.random.choice(len(non_repeating_e_src_l), size=num_smp_from_hist, replace=replace)
#
#         negative_src_l = np.concatenate([self.src_list_distinct[rnd_src_index], non_repeating_e_src_l[nre_e_index]])
#         negative_dst_l = np.concatenate([self.dst_list_distinct[rnd_dst_index], non_repeating_e_dst_l[nre_e_index]])
#
#         return negative_src_l, negative_dst_l
#
#     def reset_random_state(self):
#         self.random_state = np.random.RandomState(self.seed)


class RandEdgeSampler_adversarial(object):
    """
  Adversarial Random Edge Sampling as Negative Edges
  RandEdgeSampler_adversarial(src_list, dst_list, ts_list, last_ts_train_val, NEG_SAMPLE, seed=None, rnd_sample_ratio=0)
  """

    def __init__(self, src_list, dst_list, ts_list, last_ts_train_val, NEG_SAMPLE, seed=None, rnd_sample_ratio=0):
        """
    'src_list', 'dst_list', 'ts_list' are related to the full data! All possible edges in train, validation, test
    """
        if not (NEG_SAMPLE == 'hist_nre' or NEG_SAMPLE == 'induc_nre'):
            raise ValueError("Undefined Negative Edge Sampling Strategy!")

        self.seed = None
        self.neg_sample = NEG_SAMPLE
        self.rnd_sample_ratio = rnd_sample_ratio
        self.src_list = src_list
        self.dst_list = dst_list
        self.ts_list = ts_list
        self.src_list_distinct = np.unique(src_list)
        self.dst_list_distinct = np.unique(dst_list)
        self.ts_list_distinct = np.unique(ts_list)
        self.ts_init = min(self.ts_list_distinct)
        self.ts_end = max(self.ts_list_distinct)
        self.ts_test_split = last_ts_train_val
        self.e_train_val_l = self.get_edges_in_time_interval(self.ts_init, self.ts_test_split)

        if seed is not None:
            self.seed = seed
            np.random.seed(self.seed)
            self.random_state = np.random.RandomState(self.seed)

    def get_edges_in_time_interval(self, start_ts, end_ts):
        """
    return edges of a specific time interval
    """
        valid_ts_interval = (self.ts_list >= start_ts) * (self.ts_list <= end_ts)
        interval_src_l = self.src_list[valid_ts_interval]
        interval_dst_l = self.dst_list[valid_ts_interval]
        interval_edges = {}
        for src, dst in zip(interval_src_l, interval_dst_l):
            if (src, dst) not in interval_edges:
                interval_edges[(src, dst)] = 1
        return interval_edges

    def get_difference_edge_list(self, first_e_set, second_e_set):
        """
    return edges in the first_e_set that are not in the second_e_set
    """
        difference_e_set = set(first_e_set) - set(second_e_set)
        src_l, dst_l = [], []
        for e in difference_e_set:
            src_l.append(e[0])
            dst_l.append(e[1])
        return np.array(src_l), np.array(dst_l)

    def sample(self, size, current_split_start_ts, current_split_end_ts):
        if self.neg_sample == 'hist_nre':
            negative_src_l, negative_dst_l = self.sample_hist_NRE(size, current_split_start_ts, current_split_end_ts)
        elif self.neg_sample == 'induc_nre':
            negative_src_l, negative_dst_l = self.sample_induc_NRE(size, current_split_start_ts, current_split_end_ts)
        else:
            raise ValueError("Undefined Negative Edge Sampling Strategy!")
        return negative_src_l, negative_dst_l

    def sample_hist_NRE(self, size, current_split_start_ts, current_split_end_ts):
        """
    method one:
    "historical adversarial sampling": (~ inductive historical edges)
    randomly samples among previously seen edges that are not repeating in current batch,
    fill in any remaining with randomly sampled
    """
        history_e_dict = self.get_edges_in_time_interval(self.ts_init, current_split_start_ts)
        current_split_e_dict = self.get_edges_in_time_interval(current_split_start_ts, current_split_end_ts)
        non_repeating_e_src_l, non_repeating_e_dst_l = self.get_difference_edge_list(history_e_dict,
                                                                                     current_split_e_dict)
        num_smp_rnd = int(self.rnd_sample_ratio * size)
        num_smp_from_hist = size - num_smp_rnd
        if num_smp_from_hist > len(non_repeating_e_src_l):
            num_smp_from_hist = len(non_repeating_e_src_l)
            num_smp_rnd = size - num_smp_from_hist

        replace = len(self.src_list_distinct) < num_smp_rnd
        rnd_src_index = np.random.choice(len(self.src_list_distinct), size=num_smp_rnd, replace=replace)

        replace = len(self.dst_list_distinct) < num_smp_rnd
        rnd_dst_index = np.random.choice(len(self.dst_list_distinct), size=num_smp_rnd, replace=replace)

        replace = len(non_repeating_e_src_l) < num_smp_from_hist
        nre_e_index = np.random.choice(len(non_repeating_e_src_l), size=num_smp_from_hist, replace=replace)

        negative_src_l = np.concatenate([self.src_list_distinct[rnd_src_index], non_repeating_e_src_l[nre_e_index]])
        negative_dst_l = np.concatenate([self.dst_list_distinct[rnd_dst_index], non_repeating_e_dst_l[nre_e_index]])

        return negative_src_l, negative_dst_l

    def sample_induc_NRE(self, size, current_split_start_ts, current_split_end_ts):
        """
    method two:
    "inductive adversarial sampling": (~ inductive non repeating edges)
    considers only edges that have been seen (in red region),
    fill in any remaining with randomly sampled
    """
        history_e_dict = self.get_edges_in_time_interval(self.ts_init, current_split_start_ts)
        current_split_e_dict = self.get_edges_in_time_interval(current_split_start_ts, current_split_end_ts)
        induc_adversarial_e = set(set(history_e_dict) - set(self.e_train_val_l)) - set(current_split_e_dict)
        induc_adv_src_l, induc_adv_dst_l = [], []
        if len(induc_adversarial_e) > 0:
            for e in induc_adversarial_e:
                induc_adv_src_l.append(int(e[0]))
                induc_adv_dst_l.append(int(e[1]))
            induc_adv_src_l = np.array(induc_adv_src_l)
            induc_adv_dst_l = np.array(induc_adv_dst_l)

        num_smp_rnd = size - len(induc_adversarial_e)

        if num_smp_rnd > 0:
            replace = len(self.src_list_distinct) < num_smp_rnd
            rnd_src_index = np.random.choice(len(self.src_list_distinct), size=num_smp_rnd, replace=replace)
            replace = len(self.dst_list_distinct) < num_smp_rnd
            rnd_dst_index = np.random.choice(len(self.dst_list_distinct), size=num_smp_rnd, replace=replace)

            negative_src_l = np.concatenate([self.src_list_distinct[rnd_src_index], induc_adv_src_l])
            negative_dst_l = np.concatenate([self.dst_list_distinct[rnd_dst_index], induc_adv_dst_l])
        else:
            rnd_induc_hist_index = np.random.choice(len(induc_adversarial_e), size=size, replace=False)
            negative_src_l = induc_adv_src_l[rnd_induc_hist_index]
            negative_dst_l = induc_adv_dst_l[rnd_induc_hist_index]

        return negative_src_l, negative_dst_l

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


class RandEdgeSampler_TMC(object):
    """
    Random Edge Sampling based for Temporal Multi-Class situations
    sampled edges are from different categories considering the temporal history of the observed edges:
      - positive inductive
      - positive historical
      - negative inductive
      - negative historical
    """

    def __init__(self, src_list, dst_list, ts_list, last_ts_train_val, NEG_SAMPLE='hist_nre_MC', seed=None,
                 rnd_sample_ratio=0):
        """
        'src_list', 'dst_list', 'ts_list' are related to the full data! All possible edges in train, validation, test
        """
        if not NEG_SAMPLE == 'hist_nre_MC':
            raise ValueError("MC: Undefined Negative Edge Sampling Strategy!")

        self.seed = None
        self.neg_sample = NEG_SAMPLE
        self.rnd_sample_ratio = rnd_sample_ratio
        self.src_list = src_list
        self.dst_list = dst_list
        self.ts_list = ts_list
        self.src_list_distinct = np.unique(src_list)
        self.dst_list_distinct = np.unique(dst_list)
        self.ts_list_distinct = np.unique(ts_list)
        self.ts_init = min(self.ts_list_distinct)
        self.ts_end = max(self.ts_list_distinct)
        self.ts_test_split = last_ts_train_val
        self.e_train_val_l = self.get_edges_in_time_interval(self.ts_init, self.ts_test_split)

        if seed is not None:
            self.seed = seed
            np.random.seed(self.seed)
            self.random_state = np.random.RandomState(self.seed)

    def get_edges_in_time_interval(self, start_ts, end_ts):
        """
        return edges of a specific time interval
        """
        valid_ts_interval = (self.ts_list >= start_ts) * (self.ts_list <= end_ts)
        interval_src_l = self.src_list[valid_ts_interval]
        interval_dst_l = self.dst_list[valid_ts_interval]
        interval_edges = {}
        for src, dst in zip(interval_src_l, interval_dst_l):
            if (src, dst) not in interval_edges:
                interval_edges[(src, dst)] = 1
        return interval_edges

    def get_difference_edge_list(self, first_e_set, second_e_set):
        """
        return edges in the first_e_set that are not in the second_e_set
        """
        difference_e_set = set(first_e_set) - set(second_e_set)
        src_l, dst_l = [], []
        for e in difference_e_set:
            src_l.append(e[0])
            dst_l.append(e[1])
        return np.array(src_l), np.array(dst_l)

    def sample(self, size, current_split_start_ts, current_split_end_ts, for_test=False):
        if self.neg_sample == 'hist_nre_MC':
            neg_hist_source, neg_hist_dest, neg_rnd_source, neg_rnd_dest = self.sample_hist_NRE(size,
                                                                                                current_split_start_ts,
                                                                                                current_split_end_ts,
                                                                                                for_test)
        else:
            raise ValueError("Undefined Negative Edge Sampling Strategy!")
        return neg_hist_source, neg_hist_dest, neg_rnd_source, neg_rnd_dest

    def sample_hist_NRE(self, size, current_split_start_ts, current_split_end_ts, for_test):
        """
        NOTE: size is the total amount of random and historical negative edges
        where there are enough edges of either type, "size/2" negative edges are historical & "size/2" are random
        """
        history_e_dict = self.get_edges_in_time_interval(self.ts_init, current_split_start_ts)
        current_split_e_dict = self.get_edges_in_time_interval(current_split_start_ts, current_split_end_ts)
        not_repeating_e_src_l, not_repeating_e_dst_l = self.get_difference_edge_list(history_e_dict,
                                                                                     current_split_e_dict)
        if for_test:  # when testing the models that have been trained with diverse settings
            num_smp_rnd = int(self.rnd_sample_ratio * size)
            num_smp_from_hist = size - num_smp_rnd
            if num_smp_from_hist > len(not_repeating_e_src_l):
                num_smp_from_hist = len(not_repeating_e_src_l)
                num_smp_rnd = size - num_smp_from_hist
        else:
            num_smp_rnd = int(size / 2)
            num_smp_from_hist = size - num_smp_rnd
            if num_smp_from_hist > len(not_repeating_e_src_l):  # if there aren't enough historical edges
                num_smp_from_hist = len(not_repeating_e_src_l)
                num_smp_rnd = size - num_smp_from_hist

        replace = len(self.src_list_distinct) < num_smp_rnd
        rnd_src_index = np.random.choice(len(self.src_list_distinct), size=num_smp_rnd, replace=replace)

        replace = len(self.dst_list_distinct) < num_smp_rnd
        rnd_dst_index = np.random.choice(len(self.dst_list_distinct), size=num_smp_rnd, replace=replace)

        replace = len(not_repeating_e_src_l) < num_smp_from_hist
        nre_e_index = np.random.choice(len(not_repeating_e_src_l), size=num_smp_from_hist, replace=replace)

        neg_hist_source = not_repeating_e_src_l[nre_e_index]
        neg_hist_dest = not_repeating_e_dst_l[nre_e_index]

        neg_rnd_source = self.src_list_distinct[rnd_src_index]
        neg_rnd_dest = self.dst_list_distinct[rnd_dst_index]

        return neg_hist_source, neg_hist_dest, neg_rnd_source, neg_rnd_dest

    def get_pos_hist_and_induc_indices(self, current_split_start_ts, pos_source_l, pos_dest_l):
        """
        return the indices of the inductive positive edges
        """
        history_e_dict = self.get_edges_in_time_interval(self.ts_init, current_split_start_ts)
        pos_induc_idx_l, pos_hist_idx_l = [], []
        for idx, (src, dst) in enumerate(zip(pos_source_l, pos_dest_l)):
            if (src, dst) in history_e_dict:
                pos_hist_idx_l.append(idx)
            else:
                pos_induc_idx_l.append(idx)

        return pos_hist_idx_l, pos_induc_idx_l

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


def eval_one_epoch_original(hint, tgan, sampler, src, dst, ts, label, NUM_NEIGHBORS):
    val_ap, val_auc_roc = [], []
    measures_list = []
    with torch.no_grad():
        tgan = tgan.eval()
        # TEST_BATCH_SIZE = 30
        TEST_BATCH_SIZE = 200
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        for k in range(num_test_batch):
            # percent = 100 * k / num_test_batch
            # if k % int(0.2 * num_test_batch) == 0:
            #     logger.info('{0} progress: {1:10.4f}'.format(hint, percent))
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            # label_l_cut = label[s_idx:e_idx]

            size = len(src_l_cut)
            src_l_fake, dst_l_fake = sampler.sample(size)

            pos_prob, neg_prob = tgan.contrast_original(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)

            pred_score = np.concatenate([pos_prob.cpu().numpy(), neg_prob.cpu().numpy()])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc_roc.append(roc_auc_score(true_label, pred_score))

            # extra performance measures
            measures_dict = extra_measures(true_label, pred_score)
            measures_list.append(measures_dict)
        measures_df = pd.DataFrame(measures_list)
        avg_measures_dict = measures_df.mean()

    return np.mean(val_ap), np.mean(val_auc_roc), avg_measures_dict


def eval_one_epoch_modified(hint, tgan, sampler, src, dst, ts, label, NUM_NEIGHBORS):
    val_ap, val_auc_roc = [], []
    measures_list = []
    with torch.no_grad():
        tgan = tgan.eval()
        # TEST_BATCH_SIZE = 30
        TEST_BATCH_SIZE = 200
        num_test_instance = len(src)
        num_test_batch = int(math.ceil(num_test_instance / TEST_BATCH_SIZE))
        for k in range(num_test_batch):
            # percent = 100 * k / num_test_batch
            # if k % int(0.2 * num_test_batch) == 0:
            #     logger.info('{0} progress: {1:10.4f}'.format(hint, percent))
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            # label_l_cut = label[s_idx:e_idx]

            size = len(src_l_cut)

            if sampler.neg_sample != 'rnd':
                negative_samples_sources, negative_samples_destinations = \
                    sampler.sample(size, ts_l_cut[0], ts_l_cut[-1])
            else:
                negative_samples_sources, negative_samples_destinations = sampler.sample(size)
                negative_samples_sources = src_l_cut

            # contrast_modified(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20)
            pos_prob = tgan.contrast_modified(src_l_cut, dst_l_cut, ts_l_cut, NUM_NEIGHBORS)
            neg_prob = tgan.contrast_modified(negative_samples_sources, negative_samples_destinations, ts_l_cut,
                                              NUM_NEIGHBORS)

            pred_score = np.concatenate([pos_prob.cpu().numpy(), neg_prob.cpu().numpy()])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc_roc.append(roc_auc_score(true_label, pred_score))

            # extra performance measures
            measures_dict = extra_measures(true_label, pred_score)
            measures_list.append(measures_dict)
        measures_df = pd.DataFrame(measures_list)
        avg_measures_dict = measures_df.mean()

    return np.mean(val_ap), np.mean(val_auc_roc), avg_measures_dict


def eval_one_epoch_modified_MC(hint, tgan, sampler, src, dst, ts, label, NUM_NEIGHBORS, for_test=False):
    measures_list = []
    with torch.no_grad():
        tgan = tgan.eval()
        TEST_BATCH_SIZE = 200
        num_test_instance = len(src)
        num_test_batch = int(math.ceil(num_test_instance / TEST_BATCH_SIZE))
        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)

            # positive edges
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]

            # negative edges
            size = len(src_l_cut)
            if sampler.neg_sample == 'rnd':
                negative_samples_sources, negative_samples_destinations = sampler.sample(size)
                negative_samples_sources = src_l_cut
            elif sampler.neg_sample == 'hist_nre_MC':
                neg_hist_source, neg_hist_dest, neg_rnd_source, neg_rnd_dest = \
                    sampler.sample(size, ts_l_cut[0], ts_l_cut[-1], for_test)
                negative_samples_sources = np.concatenate([neg_hist_source, neg_rnd_source], axis=0)
                negative_samples_destinations = np.concatenate([neg_hist_dest, neg_rnd_dest], axis=0)
            else:  # hist_nre or induc_nre
                negative_samples_sources, negative_samples_destinations = \
                    sampler.sample(size, ts_l_cut[0], ts_l_cut[-1])

            pred_prob_l = []
            true_lbl_l = []
            # Negative edges
            neg_pred = tgan.contrast_modified_MC(negative_samples_sources, negative_samples_destinations,
                                                 ts_l_cut, NUM_NEIGHBORS)
            y_pred_prob_MC = torch.softmax(neg_pred, dim=1)
            pos_prob_neg_pred = multi_pred_prob_to_pos_pre_prob(y_pred_prob_MC)
            pred_prob_l.append(pos_prob_neg_pred.cpu().numpy())
            true_lbl_l.append(np.zeros(len(negative_samples_sources)))

            # Positive edges
            pos_edge_pred = tgan.contrast_modified_MC(src_l_cut, dst_l_cut, ts, NUM_NEIGHBORS)
            y_pred_prob_MC = torch.softmax(pos_edge_pred, dim=1)
            pos_prob_pos_edge_pred = multi_pred_prob_to_pos_pre_prob(y_pred_prob_MC)
            pred_prob_l.append(pos_prob_pos_edge_pred.cpu().numpy())
            true_lbl_l.append(np.ones(len(pos_prob_pos_edge_pred)))

            # concatenate different categories of edges
            pred_prob = np.concatenate(pred_prob_l)
            true_label = np.concatenate(true_lbl_l)

            # extra performance measures
            measures_dict = extra_measures(true_label, pred_prob)
            measures_list.append(measures_dict)
        measures_df = pd.DataFrame(measures_list)
        avg_measures_dict = measures_df.mean(numeric_only=True)

    return avg_measures_dict


def multi_pred_prob_to_pos_pre_prob(pred_prob):
    """
    given the multi-class prediction probabilities, returns the probability of belonging to the positive class
    NOTE: torch tensors are passed
    """
    pos_prob = []
    num_instances = pred_prob.shape[0]
    for row_idx in range(num_instances):
        # probability of being a positive edge
        pos_prob.append(pred_prob[row_idx][POS_HIST] + pred_prob[row_idx][POS_INDUC])

    pos_prob = torch.reshape(torch.tensor(pos_prob), (num_instances, 1))
    return pos_prob

def get_measures_for_threshold(y_true, y_pred_score, threshold):
    """
    compute measures for a specific threshold
    """
    perf_measures = {}
    y_pred_label = y_pred_score > threshold
    perf_measures['acc'] = accuracy_score(y_true, y_pred_label)
    prec, rec, f1, num = precision_recall_fscore_support(y_true, y_pred_label, average='binary', zero_division=1)
    perf_measures['prec'] = prec
    perf_measures['rec'] = rec
    perf_measures['f1'] = f1
    return perf_measures


def extra_measures(y_true, y_pred_score):
    """
    compute extra performance measures
    """
    perf_dict = {}
    # find optimal threshold of au-roc
    perf_dict['ap'] = average_precision_score(y_true, y_pred_score)

    perf_dict['au_roc_score'] = roc_auc_score(y_true, y_pred_score)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_score)
    opt_idx = np.argmax(tpr - fpr)
    opt_thr_auroc = roc_thresholds[opt_idx]
    perf_dict['opt_thr_au_roc'] = opt_thr_auroc
    auroc_perf_dict = get_measures_for_threshold(y_true, y_pred_score, opt_thr_auroc)
    perf_dict['acc_auroc_opt_thr'] = auroc_perf_dict['acc']
    perf_dict['prec_auroc_opt_thr'] = auroc_perf_dict['prec']
    perf_dict['rec_auroc_opt_thr'] = auroc_perf_dict['rec']
    perf_dict['f1_auroc_opt_thr'] = auroc_perf_dict['f1']

    prec_pr_curve, rec_pr_curve, pr_thresholds = precision_recall_curve(y_true, y_pred_score)
    perf_dict['au_pr_score'] = auc(rec_pr_curve, prec_pr_curve)
    # convert to f score
    fscore = (2 * prec_pr_curve * rec_pr_curve) / (prec_pr_curve + rec_pr_curve)
    opt_idx = np.argmax(fscore)
    opt_thr_aupr = pr_thresholds[opt_idx]
    perf_dict['opt_thr_au_pr'] = opt_thr_aupr
    aupr_perf_dict = get_measures_for_threshold(y_true, y_pred_score, opt_thr_aupr)
    perf_dict['acc_aupr_opt_thr'] = aupr_perf_dict['acc']
    perf_dict['prec_aupr_opt_thr'] = aupr_perf_dict['prec']
    perf_dict['rec_aupr_opt_thr'] = aupr_perf_dict['rec']
    perf_dict['f1_aupr_opt_thr'] = aupr_perf_dict['f1']

    # threshold = 0.5
    perf_half_dict = get_measures_for_threshold(y_true, y_pred_score, 0.5)
    perf_dict['acc_thr_0.5'] = perf_half_dict['acc']
    perf_dict['prec_thr_0.5'] = perf_half_dict['prec']
    perf_dict['rec_thr_0.5'] = perf_half_dict['rec']
    perf_dict['f1_thr_0.5'] = perf_half_dict['f1']

    return perf_dict

class RandEdgeSampler_CLScheduler(object):
  """
Random Edge Sampling as Negative Edges under curriculum training scheduler
RandEdgeSampler_CLScheduler(src_list, dst_list, ts_list, last_ts_train_val, NEG_SAMPLE, seed=None, num_epoch=50)
"""

  def __init__(self, src_list, dst_list, ts_list, NEG_SAMPLE, pacing="linear", seed=None, num_epoch=None, reverse_cl=False, no_proximity=False):
      """
  'src_list', 'dst_list', 'ts_list' are related to the train data! Only possible edges in train
  """
      if not (NEG_SAMPLE == 'hist_nre'):
          raise ValueError("Undefined Negative Edge Sampling Strategy!")

      self.seed = None
      self.neg_sample = NEG_SAMPLE
      self.src_list = src_list
      self.dst_list = dst_list
      self.ts_list = ts_list
      self.src_list_distinct = np.unique(src_list)
      self.dst_list_distinct = np.unique(dst_list)
      self.ts_list_distinct = np.unique(ts_list)
      self.ts_init = min(self.ts_list_distinct)
      self.ts_end = max(self.ts_list_distinct)
      self.e_train_l = self.get_edges_in_time_interval(self.ts_init, self.ts_end)
      self.num_epoch = num_epoch
      self.pacing = pacing
      self.reverse_cl = reverse_cl
      self.no_proximity = no_proximity

      if seed is not None:
          self.seed = seed
          np.random.seed(self.seed)
          self.random_state = np.random.RandomState(self.seed)

  def get_edges_in_time_interval(self, start_ts, end_ts):
      """
  return edges of a specific time interval
  """
      valid_ts_interval = (self.ts_list >= start_ts) * (self.ts_list <= end_ts)
      interval_src_l = self.src_list[valid_ts_interval]
      interval_dst_l = self.dst_list[valid_ts_interval]
      interval_edges = {}
      for src, dst in zip(interval_src_l, interval_dst_l):
          if (src, dst) not in interval_edges:
              interval_edges[(src, dst)] = 1
      return interval_edges
  
  def get_timestamps_in_time_interval(self, start_ts, end_ts):
    """
    Return timestamps of edges in a specific time interval
    """
    valid_ts_interval = (self.ts_list >= start_ts) & (self.ts_list <= end_ts)
    
    # Get source and destination nodes within the interval
    interval_src_l = self.src_list[valid_ts_interval]
    interval_dst_l = self.dst_list[valid_ts_interval]
    interval_ts_l = self.ts_list[valid_ts_interval]
    
    interval_ts_dict = {}
    for src, dst, ts in zip(interval_src_l, interval_dst_l, interval_ts_l):
        if (src, dst) not in interval_ts_dict:
            interval_ts_dict[(src, dst)] = ts
    return interval_ts_dict


  def get_difference_edge_list(self, first_e_set, second_e_set):
      """
  return edges in the first_e_set that are not in the second_e_set
  """
      difference_e_set = set(first_e_set) - set(second_e_set)
      src_l, dst_l = [], []
      for e in difference_e_set:
          src_l.append(e[0])
          dst_l.append(e[1])
      # return np.array(src_l), np.array(dst_l)
      if len(src_l)>0:
        return list(zip(src_l, dst_l))
      else:
        return []

  def sample(self, size, current_split_start_ts, current_split_end_ts, epoch):
    if self.num_epoch is None:
      g_t=0
    else:
      if self.neg_sample == 'hist_nre':
        if self.pacing=='linear':
            g_t = 1-max(1 - 0.05 - epoch/self.num_epoch, 0)
        elif self.pacing=='geometric':
            g_t = 1-max(1-np.power(2, np.log2(0.05)-np.log2(0.05)*epoch/self.num_epoch), 0)
        elif self.pacing=='root':
            g_t = 1-max(1-np.sqrt(0.05*0.05+(1-0.05*0.05)*epoch/self.num_epoch), 0)
      else:
            raise ValueError("Undefined Negative Edge Sampling Strategy!")
      if self.reverse_cl:
          g_t = 1-g_t
    negative_src_l, negative_dst_l = self.sample_hist_NRE(size, current_split_start_ts, current_split_end_ts, g_t, g_t)
    return negative_src_l, negative_dst_l

  def sample_hist_NRE_old(self, size, current_split_start_ts, current_split_end_ts, rnd_sample_ratio):
      """
  method one:
  "historical adversarial sampling"
  randomly samples among previously seen edges that are not repeating in current batch,
  fill in any remaining with randomly sampled
  """
      history_e_dict = self.get_timestamps_in_time_interval(self.ts_init, current_split_start_ts)
      current_split_e_dict = self.get_timestamps_in_time_interval(current_split_start_ts, current_split_end_ts)
      non_repeating_e_src_l, non_repeating_e_dst_l = self.get_difference_edge_list(history_e_dict,
                                                                                    current_split_e_dict)
      num_smp_rnd = int(rnd_sample_ratio * size)
      num_smp_from_hist = size - num_smp_rnd
      if num_smp_from_hist > len(non_repeating_e_src_l):
          num_smp_from_hist = len(non_repeating_e_src_l)
          num_smp_rnd = size - num_smp_from_hist

      replace = len(self.src_list_distinct) < num_smp_rnd
      rnd_src_index = np.random.choice(len(self.src_list_distinct), size=num_smp_rnd, replace=replace)

      replace = len(self.dst_list_distinct) < num_smp_rnd
      rnd_dst_index = np.random.choice(len(self.dst_list_distinct), size=num_smp_rnd, replace=replace)

      replace = len(non_repeating_e_src_l) < num_smp_from_hist
      nre_e_index = np.random.choice(len(non_repeating_e_src_l), size=num_smp_from_hist, replace=replace)

      negative_src_l = np.concatenate([self.src_list_distinct[rnd_src_index], non_repeating_e_src_l[nre_e_index]])
      negative_dst_l = np.concatenate([self.dst_list_distinct[rnd_dst_index], non_repeating_e_dst_l[nre_e_index]])

      return negative_src_l, negative_dst_l
  
  def sample_hist_NRE(self, size, current_split_start_ts, current_split_end_ts, H_t, D_t):
    """
    "historical adversarial sampling" with given H(t) and D(t)
    """
    history_e_dict = self.get_timestamps_in_time_interval(self.ts_init, current_split_start_ts)
    current_split_e_dict = self.get_timestamps_in_time_interval(current_split_start_ts, current_split_end_ts)
    non_repeating_e = self.get_difference_edge_list(history_e_dict, current_split_e_dict)

    if len(non_repeating_e) == 0:
        # If there are no non-repeating edges, simply random sample nodes
        replace_src = len(self.src_list_distinct) < size
        negative_src_l = np.random.choice(self.src_list_distinct, size, replace=replace_src)

        replace_dst = len(self.dst_list_distinct) < size
        negative_dst_l = np.random.choice(self.dst_list_distinct, size, replace=replace_dst)
        return negative_src_l, negative_dst_l

    # Calculate D_i for all historical edges
    distance_to_current = current_split_start_ts - np.array([history_e_dict[key] for key in non_repeating_e])
    if len(set(distance_to_current))<=1:
      weights = [1.0 for _ in distance_to_current]
    else:
      D_max = np.max(distance_to_current)
      D_min = np.min(distance_to_current)

      normalized_distances = (distance_to_current - D_min) / (D_max - D_min)
      
      # Compute weights based on normalized distances
      weights = 1 - np.abs((1-D_t) - normalized_distances)
      # Ensure the weights are within [0, 1]
      weights = np.clip(weights, 0, 1)

      # Resize the weights to match non_repeating_e_src_l length
      # weights = weights[:len(non_repeating_e_src_l)]
    weights = np.array(weights)
    weights /= weights.sum()

    num_smp_rnd = int((1 - H_t) * size)
    num_smp_from_hist = size - num_smp_rnd

    # Sample from historical edges considering D(t)
    if not self.no_proximity:
        nre_e_index = np.random.choice(len(non_repeating_e), size=num_smp_from_hist, replace=True, p=weights)
    else:
        nre_e_index = np.random.choice(len(non_repeating_e), size=num_smp_from_hist, replace=True)



    # replace_src = len(self.src_list_distinct) < num_smp_rnd
    # rnd_src_index = np.random.choice(len(self.src_list_distinct), size=num_smp_rnd, replace=replace_src)

    replace_dst = len(self.dst_list_distinct) < num_smp_rnd
    rnd_dst_index = np.random.choice(len(self.dst_list_distinct), size=num_smp_rnd, replace=replace_dst)

    # negative_src_l = np.concatenate([self.src_list_distinct[rnd_src_index], non_repeating_e_src_l[nre_e_index]])
    _, non_repeating_e_dst = list(map(list, zip(*non_repeating_e)))
    non_repeating_e_dst = np.array(non_repeating_e_dst)
    negative_dst_l = np.concatenate([self.dst_list_distinct[rnd_dst_index], non_repeating_e_dst[nre_e_index]])

    return None, negative_dst_l
