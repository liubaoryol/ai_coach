TUTORIAL_MAP = {
    "x_grid": 5,
    "y_grid": 5,
    "a1_init": (4, 1),
    "a2_init": (4, 3),
    "boxes": [(0, 0), (0, 4), (2, 4)],
    "goals": [(4, 2)],
    "walls": [(1, i + 1) for i in range(3)],
    "wall_dir": [1 for dummy_i in range(3)],
    "drops": []
}

EXP1_MAP = {
    "name":
    "exp1",
    "x_grid":
    7,
    "y_grid":
    7,
    "a1_init": (6, 2),
    "a2_init": (6, 4),
    "boxes": [(0, 0), (1, 3), (0, 6)],
    "goals": [(6, 3)],
    "walls": [(3, 0), (4, 0), (1, 2), (2, 2), (2, 3), (2, 4), (3, 6), (4, 6),
              (5, 2), (5, 3), (5, 4)],
    "wall_dir": [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1],
    "drops": []
}

TEST_MAP = {
    "name": "test1",
    "x_grid": 3,
    "y_grid": 3,
    "a1_init": (0, 1),
    "a2_init": (0, 2),
    "boxes": [(0, 0), (1, 2)],
    "goals": [(0, 2)],
    "walls": [(1, 1), (2, 1), (2, 2)],
    "wall_dir": [0, 0, 1],
    "drops": []
}

TEST_MAP2 = {
    "name": "test2",
    "x_grid": 2,
    "y_grid": 2,
    "a1_init": (1, 0),
    "a2_init": (1, 1),
    "boxes": [(0, 0)],
    "goals": [(0, 1)],
    "walls": [],
    "wall_dir": [],
    "drops": []
}

TEST_MAP3 = {
    "x_grid": 2,
    "y_grid": 1,
    "a1_init": (0, 0),
    "a2_init": (0, 0),
    "boxes": [(1, 0)],
    "goals": [(0, 0)],
    "walls": [],
    "wall_dir": [],
    "drops": []
}
