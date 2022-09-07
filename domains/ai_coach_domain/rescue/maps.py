from ai_coach_domain.rescue import Route, Location, E_Type, Work, Place

MAP_RESCUE = {
    "places": [
        Place("Fire Station", (0.4, 0.4)),
        Place("City Hall", (0.1, 0.1), helps=1),
        Place("Police Station", (0.1, 0.9)),
        Place("Bridge", (0.8, 0.8)),
        Place("Campsite", (0.9, 0.1), helps=2),
        Place("Mall", (0.9, 0.9), helps=4),
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
        Place("Fire Station", (0.4, 0.4)),
        Place("City Hall", (0.1, 0.1), helps=1),
        Place("Police Station", (0.1, 0.9)),
        Place("Bridge 1", (0.75, 0.85)),
        Place("Campsite", (0.9, 0.1), helps=2),
        Place("Mall", (0.9, 0.9), helps=4),
        Place("Bridge 2", (0.85, 0.75)),
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
