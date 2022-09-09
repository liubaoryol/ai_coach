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
        Place(PlaceName.Fire_stateion, (0.44, 0.35)),
        Place(PlaceName.City_hall, (0.12, 0.1), helps=1),
        Place(PlaceName.Police_station, (0.12, 0.52)),
        Place(PlaceName.Bridge_1, (0.58, 0.82)),
        Place(PlaceName.Campsite, (0.85, 0.11), helps=2),
        Place(PlaceName.Mall, (0.9, 0.92), helps=4),
        Place(PlaceName.Bridge_2, (0.91, 0.50)),
    ],
    "routes": [
        Route(start=0, end=1, length=4),
        Route(start=0, end=2, length=2),
        Route(start=0, end=6, length=4),
        Route(start=0, end=4, length=5),
        Route(start=2, end=3, length=5),
        Route(start=2, end=1, length=6),
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
