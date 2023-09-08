import ast
import numpy as np
import click


@click.command()
@click.option("--filepath", required=True, type=str, help="Path to log file")
def main(filepath):
  dict_data = {}
  with open(filepath) as f:
    lines = f.readlines()
    for idx, row in enumerate(lines):
      pidx = row.find('{')
      if pidx != -1:
        alg_name = None
        is_no_tx_semi = False
        for idx2 in range(1, 10):
          char_idx = lines[idx - idx2].find("#########")
          if char_idx != -1:
            alg_row = lines[idx - idx2 - 1]
            alg_name = alg_row[char_idx:-1]
            if alg_name[0:4] == 'Semi' and alg_row.find('_aws:') == -1:
              info_row = lines[idx - idx2 + 1]
              if info_row.find('without') != -1:
                is_no_tx_semi = True

            break
        fidx = 1 if not is_no_tx_semi else 3
        x_row = lines[idx - fidx]
        x_data_string = x_row[char_idx:-1]
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
    # a1 x mean, a1 x std, a2 x mean, a2 x std,
    # a1 wJS mean, a1 wJS std, a2 wJS mean, a2 wJS std
    str_res = ""
    str_res += "%.6f," % dict_results[key]['x mean'][0][0]
    str_res += "%.6f," % dict_results[key]['x mean'][1][0]
    str_res += "%.6f," % dict_results[key]['x mean'][0][1]
    str_res += "%.6f," % dict_results[key]['x mean'][1][1]
    str_res += "%.6f," % dict_results[key]['wJS'][0][0]
    str_res += "%.6f," % dict_results[key]['wJS'][1][0]
    str_res += "%.6f," % dict_results[key]['wJS'][0][1]
    str_res += "%.6f" % dict_results[key]['wJS'][1][1]

    print(str_res)


if __name__ == "__main__":
  main()
