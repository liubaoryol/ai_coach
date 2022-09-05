from ai_coach_domain.rescue import Route, Location, E_Type, Work, Place


class RescueName:
  FireStation = "Fire"
  PoliceStation = "Police"
  CityHall = "CityHall"
  Bridge = "Bridge"
  Campsite = "Camp"
  Mall = "Mall"


MAP_RESCUE = {
    "routes": [
        Route(RescueName.FireStation, RescueName.CityHall, 4),
        Route(RescueName.FireStation, RescueName.PoliceStation, 2),
        Route(RescueName.FireStation, RescueName.Bridge, 4),
        Route(RescueName.FireStation, RescueName.Campsite, 5),
        Route(RescueName.PoliceStation, RescueName.Bridge, 5),
        Route(RescueName.PoliceStation, RescueName.CityHall, 6),
    ],
    "places": {
        RescueName.FireStation:
        Place((0.4, 0.4), [(E_Type.Route, 0), (E_Type.Route, 1),
                           (E_Type.Route, 2), (E_Type.Route, 3)]),
        RescueName.CityHall:
        Place((0.1, 0.1), [(E_Type.Route, 0), (E_Type.Route, 5)]),
        RescueName.PoliceStation:
        Place((0.1, 0.9), [(E_Type.Route, 5), (E_Type.Route, 1),
                           (E_Type.Route, 4)]),
        RescueName.Bridge:
        Place((0.8, 0.8), [(E_Type.Route, 2), (E_Type.Route, 4)]),
        RescueName.Campsite:
        Place((0.9, 0.1), [(E_Type.Route, 3)]),
        RescueName.Mall:
        Place((0.9, 0.9), []),
    },
    "visible_places": [
        RescueName.FireStation, RescueName.CityHall, RescueName.PoliceStation,
        RescueName.Bridge, RescueName.Campsite
    ],
    "work_locations": [
        Location(E_Type.Place, RescueName.CityHall),
        Location(E_Type.Place, RescueName.Bridge),
        Location(E_Type.Place, RescueName.Campsite)
    ],
    "work_info": [Work(1, 1), Work(4, 2), Work(2, 1)],
    "a1_init":
    Location(E_Type.Place, RescueName.PoliceStation),
    "a2_init":
    Location(E_Type.Place, RescueName.FireStation),
}
