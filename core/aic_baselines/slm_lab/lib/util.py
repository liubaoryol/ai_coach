from collections import deque
import collections.abc
from datetime import datetime
from importlib import reload
import json
import numpy as np
import operator
import os
import pandas as pd
import pickle
import sys
import time
import torch
import torch.multiprocessing as mp
import yaml

NUM_CPUS = mp.cpu_count()
FILE_TS_FORMAT = '%Y_%m_%d_%H%M%S'


class LabJsonEncoder(json.JSONEncoder):

  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    elif isinstance(obj, np.floating):
      return float(obj)
    elif isinstance(obj, (np.ndarray, pd.Series)):
      return obj.tolist()
    else:
      return str(obj)


def batch_get(arr, idxs):
  '''Get multi-idxs from an array depending if it's a python list or np.array'''
  if isinstance(arr, (list, deque)):
    return np.array(operator.itemgetter(*idxs)(arr))
  else:
    return arr[idxs]


def calc_srs_mean_std(sr_list):
  '''Given a list of series, calculate their mean and std'''
  cat_df = pd.DataFrame(dict(enumerate(sr_list)))
  mean_sr = cat_df.mean(axis=1)
  std_sr = cat_df.std(axis=1)
  return mean_sr, std_sr


def calc_ts_diff(ts2, ts1):
  '''
    Calculate the time from tss ts1 to ts2
    @param {str} ts2 Later ts in the FILE_TS_FORMAT
    @param {str} ts1 Earlier ts in the FILE_TS_FORMAT
    @returns {str} delta_t in %H:%M:%S format
    @example

    ts1 = '2017_10_17_084739'
    ts2 = '2017_10_17_084740'
    ts_diff = util.calc_ts_diff(ts2, ts1)
    # => '0:00:01'
    '''
  delta_t = datetime.strptime(ts2, FILE_TS_FORMAT) - datetime.strptime(
      ts1, FILE_TS_FORMAT)
  return str(delta_t)


def cast_df(val):
  '''missing pydash method to cast value as DataFrame'''
  if isinstance(val, pd.DataFrame):
    return val
  return pd.DataFrame(val)


def concat_batches(batches):
  '''
    Concat batch objects from body.memory.sample() into one batch, 
    when all bodies experience similar envs
    Also concat any nested epi sub-batches into flat batch
    {k: arr1} + {k: arr2} = {k: arr1 + arr2}
    '''
  # if is nested, then is episodic
  is_episodic = isinstance(batches[0]['dones'][0], (list, np.ndarray))
  concat_batch = {}
  for k in batches[0]:
    datas = []
    for batch in batches:
      data = batch[k]
      if is_episodic:  # make into plain batch instead of nested
        data = np.concatenate(data)
      datas.append(data)
    concat_batch[k] = np.concatenate(datas)
  return concat_batch


def downcast_float32(df):
  '''Downcast any float64 col to float32 to allow safer pandas comparison'''
  for col in df.columns:
    if df[col].dtype == 'float':
      df[col] = df[col].astype('float32')
  return df


def epi_done(done):
  '''
  General method to check if episode is done for both single and vectorized env
  Only return True for singleton done since vectorized env does not have 
  a natural episode boundary
  '''
  return np.isscalar(done) and done


def frame_mod(frame, frequency, num_envs):
  '''
  Generic mod for (frame % frequency == 0) for when num_envs is 1 or more,
  since frame will increase multiple ticks for vector env, use the remainder
  '''
  remainder = num_envs or 1
  return (frame % frequency < remainder)


def get_class_name(obj, lower=False):
  '''Get the class name of an object'''
  class_name = obj.__class__.__name__
  if lower:
    class_name = class_name.lower()
  return class_name


def get_file_ext(data_path):
  '''get the `.ext` of file.ext'''
  return os.path.splitext(data_path)[-1]


def get_lab_mode():
  return os.environ.get('lab_mode')


def get_prepath(spec, unit='experiment'):
  spec_name = spec['name']
  meta_spec = spec['meta']
  predir = f'data/{spec_name}_{meta_spec["experiment_ts"]}'
  prename = f'{spec_name}'
  trial_index = meta_spec['trial']
  session_index = meta_spec['session']
  t_str = '' if trial_index is None else f'_t{trial_index}'
  s_str = '' if session_index is None else f'_s{session_index}'
  if unit == 'trial':
    prename += t_str
  elif unit == 'session':
    prename += f'{t_str}{s_str}'
  prepath = f'{predir}/{prename}'
  return prepath


def get_session_df_path(session_spec, df_mode):
  '''Method to return standard filepath for session_df 
  (agent.body.train_df/eval_df) for saving and loading'''
  info_prepath = session_spec['meta']['info_prepath']
  return f'{info_prepath}_session_df_{df_mode}.csv'


def insert_folder(prepath, folder):
  '''Insert a folder into prepath'''
  split_path = prepath.split('/')
  prename = split_path.pop()
  split_path += [folder, prename]
  return '/'.join(split_path)


def parallelize(fn, args, num_cpus=NUM_CPUS):
  '''
  Parallelize a method fn, args and return results with order preserved per args
  args should be a list of tuples.
  @returns {list} results Order preserved output from fn.
  '''
  pool = mp.Pool(num_cpus, maxtasksperchild=1)
  results = pool.starmap(fn, args)
  pool.close()
  pool.join()
  return results


def read_as_df(data_path, **kwargs):
  '''Submethod to read data as DataFrame'''
  data = pd.read_csv(data_path, **kwargs)
  return data


def read_as_pickle(data_path, **kwargs):
  '''Submethod to read data as pickle'''
  with open(data_path, 'rb') as f:
    data = pickle.load(f)
  return data


def set_cuda_id(spec):
  '''
  Use trial and session id to hash and modulo cuda device count for a cuda_id
  to maximize device usage. Sets the net_spec for the base Net class to pick up
  '''
  # Don't trigger any cuda call if not using GPU. Otherwise will break
  # multiprocessing on machines with CUDA.
  # see issues https://github.com/pytorch/pytorch/issues/334
  # https://github.com/pytorch/pytorch/issues/3491
  # https://github.com/pytorch/pytorch/issues/9996
  for agent_spec in spec['agent']:
    if not agent_spec['net'].get('gpu'):
      return
  meta_spec = spec['meta']
  trial_idx = meta_spec['trial'] or 0
  session_idx = meta_spec['session'] or 0
  # shared hogwild uses only global networks, offset them to idx 0
  if meta_spec['distributed'] == 'shared':
    session_idx = 0
  job_idx = trial_idx * meta_spec['max_session'] + session_idx
  job_idx += meta_spec['cuda_offset']
  device_count = torch.cuda.device_count()
  cuda_id = job_idx % device_count if torch.cuda.is_available() else None

  for agent_spec in spec['agent']:
    agent_spec['net']['cuda_id'] = cuda_id


def set_logger(spec, logger, unit=None):
  '''Set the logger for a lab unit give its spec'''
  os.environ['LOG_PREPATH'] = insert_folder(get_prepath(spec, unit=unit), 'log')
  reload(logger)  # to set session-specific logger


def set_random_seed(spec):
  '''Generate and set random seed for relevant modules, 
  and record it in spec.meta.random_seed'''
  trial = spec['meta']['trial']
  session = spec['meta']['session']
  random_seed = int(1e5 * (trial or 0) + 1e3 * (session or 0) + time.time())
  torch.cuda.manual_seed_all(random_seed)
  torch.manual_seed(random_seed)
  np.random.seed(random_seed)
  spec['meta']['random_seed'] = random_seed
  return random_seed


def _sizeof(obj, seen=None):
  '''Recursively finds size of objects'''
  size = sys.getsizeof(obj)
  if seen is None:
    seen = set()
  obj_id = id(obj)
  if obj_id in seen:
    return 0
  # Important mark as seen *before* entering recursion to gracefully handle
  # self-referential objects
  seen.add(obj_id)
  if isinstance(obj, dict):
    size += sum([_sizeof(v, seen) for v in obj.values()])
    size += sum([_sizeof(k, seen) for k in obj.keys()])
  elif hasattr(obj, '__dict__'):
    size += _sizeof(obj.__dict__, seen)
  elif hasattr(obj, '__iter__') and not isinstance(obj,
                                                   (str, bytes, bytearray)):
    size += sum([_sizeof(i, seen) for i in obj])
  return size


def sizeof(obj, divisor=1e6):
  '''Return the size of object, in MB by default'''
  return _sizeof(obj) / divisor


def split_minibatch(batch, mb_size):
  '''Split a batch into minibatches of mb_size or smaller, without replacement
  '''
  size = len(batch['rewards'])
  assert mb_size < size, f'Minibatch size {mb_size} must be < batch size {size}'
  idxs = np.arange(size)
  np.random.shuffle(idxs)
  chunks = int(size / mb_size)
  nested_idxs = np.array_split(idxs[:chunks * mb_size], chunks)
  if size % mb_size != 0:  # append leftover from split
    nested_idxs += [idxs[chunks * mb_size:]]
  mini_batches = []
  for minibatch_idxs in nested_idxs:
    minibatch = {k: v[minibatch_idxs] for k, v in batch.items()}
    mini_batches.append(minibatch)
  return mini_batches


def to_json(d, indent=2):
  '''Shorthand method for stringify JSON with indent'''
  return json.dumps(d, indent=indent, cls=LabJsonEncoder)


def to_render():
  return os.environ.get(
      'RENDER',
      'false') == 'true' or (get_lab_mode() in ('dev', 'enjoy')
                             and os.environ.get('RENDER', 'true') == 'true')


def write_as_df(data, data_path):
  '''Submethod to write data as DataFrame'''
  df = cast_df(data)
  df.to_csv(data_path, index=False)
  return data_path


def write_as_pickle(data, data_path):
  '''Submethod to write data as pickle'''
  with open(data_path, 'wb') as f:
    pickle.dump(data, f)
  return data_path


def write_as_plain(data, data_path):
  '''Submethod to write data as plain type'''
  open_file = open(data_path, 'w')
  ext = get_file_ext(data_path)
  if ext == '.json':
    json.dump(data, open_file, indent=2, cls=LabJsonEncoder)
  elif ext == '.yml':
    yaml.dump(data, open_file)
  else:
    open_file.write(str(data))
  open_file.close()
  return data_path


def to_torch_batch(batch, device, is_episodic):
  '''Mutate a batch (dict) to make its values from numpy into PyTorch tensor'''
  for k in batch:
    if is_episodic:  # for episodic format
      batch[k] = np.concatenate(batch[k])
    elif isinstance(batch[k], collections.abc.Sequence):
      # elif ps.is_list(batch[k]):
      batch[k] = np.array(batch[k])
    batch[k] = torch.from_numpy(batch[k].astype(np.float32)).to(device)
  return batch
