class Scene:
    def __init__(self, camera, shapes, bsdfs, mediums, phases, area_lights):
        self.camera = camera
        self.shapes = shapes
        self.bsdfs = bsdfs
        self.area_lights = area_lights
        self.mediums = mediums
        self.phases = phases
