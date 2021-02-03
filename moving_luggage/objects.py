
class ObjCommon():
    def __init__(self, coord):
        self.coord = coord


class ObjAgent(ObjCommon):
    def __init__(self, coord):
        super().__init__(coord)
        self.hold = False
        self.id = None
