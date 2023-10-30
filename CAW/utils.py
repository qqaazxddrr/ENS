import numpy as np
import torch
import os
import random


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
        self.epoch_count += 1
        return self.num_round >= self.max_round


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list, seed=None):
        self.seed = None
        self.neg_sample = 'rnd'
        src_list = np.concatenate(src_list)
        dst_list = np.concatenate(dst_list)
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
#         'src_list', 'dst_list', 'ts_list' are related to the full data! All possible edges in train, validation, test
#         """
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
#         return edges of a specific time interval
#         """
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
#         return edges in the first_e_set that are not in the second_e_set
#         """
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
                induc_adv_src_l.append(e[0])
                induc_adv_dst_l.append(e[1])
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

  def __init__(self, src_list, dst_list, ts_list, last_ts_train_val, NEG_SAMPLE='hist_nre_MC', seed=None, rnd_sample_ratio=0):
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
      num_smp_rnd = int(size/2)
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
    neg_rnd_dest =  self.dst_list_distinct[rnd_dst_index]

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



def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def process_sampling_numbers(num_neighbors, num_layers):
    num_neighbors = [int(n) for n in num_neighbors]
    if len(num_neighbors) == 1:
        num_neighbors = num_neighbors * num_layers
    else:
        num_layers = len(num_neighbors)
    return num_neighbors, num_layers

class RandEdgeSampler_CLScheduler(object):
  """
Random Edge Sampling as Negative Edges under curriculum training scheduler
RandEdgeSampler_CLScheduler(src_list, dst_list, ts_list, last_ts_train_val, NEG_SAMPLE, seed=None, num_epoch=50)
"""

  def __init__(self, src_list, dst_list, ts_list, NEG_SAMPLE, pacing="linear", seed=None, num_epoch=None):
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
        replace_dst = len(self.dst_list_distinct) < size
        negative_dst_l = np.random.choice(self.dst_list_distinct, size, replace=replace_dst)
        return None, negative_dst_l

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
    nre_e_index = np.random.choice(len(non_repeating_e), size=num_smp_from_hist, replace=True, p=weights)

    # replace_src = len(self.src_list_distinct) < num_smp_rnd
    # rnd_src_index = np.random.choice(len(self.src_list_distinct), size=num_smp_rnd, replace=replace_src)

    replace_dst = len(self.dst_list_distinct) < num_smp_rnd
    rnd_dst_index = np.random.choice(len(self.dst_list_distinct), size=num_smp_rnd, replace=replace_dst)

    # negative_src_l = np.concatenate([self.src_list_distinct[rnd_src_index], non_repeating_e_src_l[nre_e_index]])
    _, non_repeating_e_dst = list(map(list, zip(*non_repeating_e)))
    non_repeating_e_dst = np.array(non_repeating_e_dst)
    negative_dst_l = np.concatenate([self.dst_list_distinct[rnd_dst_index], non_repeating_e_dst[nre_e_index]])

    return None, negative_dst_l