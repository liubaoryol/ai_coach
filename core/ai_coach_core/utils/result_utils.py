def norm_hamming_distance(seq1, seq2):
  assert len(seq1) == len(seq2)

  count = 0
  for idx, elem in enumerate(seq1):
    if elem != seq2[idx]:
      count += 1

  return count / len(seq1)


def alignment_sequence(seq1, seq2):
  assert len(seq1) == len(seq2)

  seq_align = []
  for idx in range(len(seq1)):
    if seq1[idx] == seq2[idx]:
      seq_align.append(1)
    else:
      seq_align.append(0)

  return seq_align
