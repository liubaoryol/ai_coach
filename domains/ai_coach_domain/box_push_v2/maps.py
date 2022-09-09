MAP_MOVERS = {
    "name":
    "movers",
    "x_grid":
    7,
    "y_grid":
    7,
    "a1_init": (2, 3),
    "a2_init": (4, 3),
    "boxes": [(0, 0), (2, 6), (6, 6)],
    "box_types": [2, 2, 2],  # number of agents needed to pick up each box
    "goals": [(3, 3)],
    "walls": [(4, 0), (5, 0), (6, 0), (5, 1), (6, 1), (2, 2), (5, 2), (6, 2),
              (4, 4), (2, 5), (5, 6)],
    "wall_dir": [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1],
    "drops": []
}

MAP_CLEANUP = {
    "name":
    "cleanup",
    "x_grid":
    7,
    "y_grid":
    7,
    "a1_init": (2, 3),
    "a2_init": (4, 3),
    "boxes": [(2, 1), (5, 0), (4, 5), (1, 6)],
    "box_types": [1, 1, 1, 1],  # number of agents needed to pick up each box
    "goals": [(3, 3)],
    "walls": [(0, 0), (1, 0), (4, 0), (0, 1), (4, 1), (5, 1), (0, 2), (2, 2),
              (0, 3), (6, 3), (4, 4), (6, 4), (1, 5), (2, 5), (6, 5), (2, 6),
              (5, 6), (6, 6)],
    "wall_dir": [0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0],
    "drops": []
}
