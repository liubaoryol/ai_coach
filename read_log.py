import os
import ast
import numpy as np

if __name__ == "__main__":
  # filename = "team_w_tx_box_push_dynamic_results.log"
  # filename = "team_wo_tx_dynamic_results.log"
  # filename = "indv_wo_tx_dynamic_results.log"
  # filename = "indv_w_tx_box_push_dynamic_results.log"
  # filename = "box_push_dynamic_results_w_tx_indv.log"
  #   filename = "bp_aws_indv.log"
  filename = "box_push_aws_results_team2.log"

  dict_data = {}
  with open(filename) as f:
    lines = f.readlines()
    for idx, row in enumerate(lines):
      pidx = row.find('{')
      if pidx != -1:
        alg_name = None
        is_no_tx_semi = False
        for idx2 in range(1, 10):
          if lines[idx - idx2].find("#########") != -1:
            alg_row = lines[idx - idx2 - 1]
            sidx = alg_row.find('ush2:')
            if sidx == -1:
              sidx = alg_row.find('_aws:')
            if sidx == -1:
              sidx = alg_row.find('push:')
            alg_name = alg_row[sidx + 6:-1]
            if alg_name[0:4] == 'Semi' and alg_row.find('_aws:') == -1:
              info_row = lines[idx - idx2 + 1]
              if info_row.find('without') != -1:
                is_no_tx_semi = True

            break
        fidx = 1 if not is_no_tx_semi else 3
        x_row = lines[idx - fidx]
        xidx = x_row.find('ush2:')
        if xidx == -1:
          xidx = x_row.find('_aws:')
        if xidx == -1:
          xidx = x_row.find('push:')
        x_data_string = x_row[xidx + 6:-1]
        x_data = [float(num) for num in x_data_string.split(',')]
        x_data = {'x mean': [x_data[0], x_data[2]]}

        p_data = ast.literal_eval(row[pidx:-1])
        x_data.update(p_data)

        if alg_name in dict_data:
          dict_data[alg_name].append(x_data)
        else:
          dict_data[alg_name] = [x_data]

  dict_compiled = {}
  for alg in dict_data:
    dict_compiled[alg] = {}
    print(len(dict_data[alg]))
    for idx, res in enumerate(dict_data[alg]):
      for key in res:
        if key not in dict_compiled[alg]:
          dict_compiled[alg][key] = np.zeros((len(dict_data[alg]), 2))
          dict_compiled[alg][key][idx] = res[key]
        else:
          dict_compiled[alg][key][idx] = res[key]

  dict_results = {}
  for alg in dict_compiled:
    dict_results[alg] = {}
    dict_results[alg]['wJS'] = [[0, 0], [0, 0]]
    data = dict_compiled[alg]
    for key in data:
      mean_data = data[key].mean(axis=0)
      std_data = data[key].std(axis=0)
      dict_results[alg][key] = [mean_data, std_data]

  for key in dict_results:
    print(key)
    str_res = (str(dict_results[key]['x mean'][0][0]) + ',' +
               str(dict_results[key]['x mean'][1][0]) + ',' +
               str(dict_results[key]['x mean'][0][1]) + ',' +
               str(dict_results[key]['x mean'][1][1]) + ',' +
               str(dict_results[key]['wJS'][0][0]) + ',' +
               str(dict_results[key]['wJS'][1][0]) + ',' +
               str(dict_results[key]['wJS'][0][1]) + ',' +
               str(dict_results[key]['wJS'][1][1]))
    print(str_res)
