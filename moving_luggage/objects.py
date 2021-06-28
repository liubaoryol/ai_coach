from moving_luggage.constants import UNKNOWN_LATENT

class ObjCommon():
    def __init__(self, coord):
        self.coord = tuple(coord)


class ObjAgent(ObjCommon):
    def __init__(self, coord):
        super().__init__(coord)
        self.hold = False
        self.id = None
        self.latent = UNKNOWN_LATENT
