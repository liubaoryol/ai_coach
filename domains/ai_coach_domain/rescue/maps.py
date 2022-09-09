from ai_coach_domain.rescue import (Route, Location, E_Type, Work, Place,
                                    PlaceName)

MAP_RESCUE = {
    "places": [
        Place(PlaceName.Fire_stateion, (0.4, 0.4)),
        Place(PlaceName.City_hall, (0.1, 0.1), helps=1),
        Place(PlaceName.Police_station, (0.1, 0.9)),
        Place(PlaceName.Bridge_1, (0.8, 0.8)),
        Place(PlaceName.Campsite, (0.9, 0.1), helps=2),
        Place(PlaceName.Mall, (0.9, 0.9), helps=4),
    ],
    "routes": [
        Route(start=0, end=1, length=4),
        Route(start=0, end=2, length=2),
        Route(start=0, end=3, length=4),
        Route(start=0, end=4, length=5),
        Route(start=2, end=3, length=5),
        Route(start=2, end=1, length=6),
    ],
    "connections": {
        0: [(E_Type.Route, 0), (E_Type.Route, 1), (E_Type.Route, 2),
            (E_Type.Route, 3)],
        1: [(E_Type.Route, 0), (E_Type.Route, 5)],
        2: [(E_Type.Route, 5), (E_Type.Route, 1), (E_Type.Route, 4)],
        3: [(E_Type.Route, 2), (E_Type.Route, 4)],
        4: [(E_Type.Route, 3)],
        5: [],
    },
    "work_locations": [
        Location(E_Type.Place, id=1),
        Location(E_Type.Place, id=3),
        Location(E_Type.Place, id=4)
    ],
    "work_info": [
        Work(workload=1, coupled_works=[]),
        Work(workload=2, coupled_works=[]),
        Work(workload=1, coupled_works=[])
    ],
    "a1_init":
    Location(E_Type.Place, 2),
    "a2_init":
    Location(E_Type.Place, 0),
}

MAP_RESCUE_2 = {
    "places": [
        Place(PlaceName.Fire_stateion, (0.41, 0.4)),
        Place(PlaceName.City_hall, (0.17, 0.08), helps=1),
        Place(PlaceName.Police_station, (0.21, 0.57)),
        Place(PlaceName.Bridge_1, (0.67, 0.86)),
        Place(PlaceName.Campsite, (0.84, 0.11), helps=2),
        Place(PlaceName.Mall, (0.92, 0.93), helps=4),
        Place(PlaceName.Bridge_2, (0.8, 0.7)),
    ],
    "routes": [
        Route(start=0,
              end=1,
              length=4,
              coords=[(0.36, 0.3), (0.33, 0.2), (0.32, 0.12), (0.26, 0.06)]),
        Route(start=0, end=2, length=2, coords=[(0.36, 0.48), (0.3, 0.54)]),
        Route(start=0,
              end=6,
              length=4,
              coords=[(0.48, 0.48), (0.53, 0.52), (0.6, 0.6), (0.7, 0.65)]),
        Route(start=0,
              end=4,
              length=5,
              coords=[(0.52, 0.39), (0.61, 0.36), (0.72, 0.32), (0.81, 0.28),
                      (0.83, 0.2)]),
        Route(start=2,
              end=3,
              length=5,
              coords=[(0.25, 0.65), (0.31, 0.7), (0.38, 0.78), (0.46, 0.83),
                      (0.56, 0.86)]),
        Route(start=2,
              end=1,
              length=6,
              coords=[(0.22, 0.5), (0.21, 0.42), (0.14, 0.37), (0.09, 0.31),
                      (0.1, 0.25), (0.12, 0.18)]),
    ],
    "connections": {
        0: [(E_Type.Route, 0), (E_Type.Route, 1), (E_Type.Route, 2),
            (E_Type.Route, 3)],
        1: [(E_Type.Route, 0), (E_Type.Route, 5)],
        2: [(E_Type.Route, 5), (E_Type.Route, 1), (E_Type.Route, 4)],
        3: [(E_Type.Route, 4)],
        4: [(E_Type.Route, 3)],
        5: [],
        6: [(E_Type.Route, 2)],
    },
    "work_locations": [
        Location(E_Type.Place, id=1),
        Location(E_Type.Place, id=3),
        Location(E_Type.Place, id=4),
        Location(E_Type.Place, id=6),
    ],
    "work_info": [
        Work(workload=1),
        Work(workload=2, coupled_works=[3]),
        Work(workload=1),
        Work(workload=2, coupled_works=[1]),
    ],
    "a1_init":
    Location(E_Type.Place, 2),
    "a2_init":
    Location(E_Type.Place, 0),
}
