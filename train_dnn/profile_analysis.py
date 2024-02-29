import cProfile
import pstats
import run_algs  # noqa: F401

if __name__ == "__main__":
  cProfile.runctx('run_algs.main()', globals(), locals(), filename='profile')

  p = pstats.Stats('profile')
  p.strip_dirs().sort_stats('cumulative').print_stats(200)
