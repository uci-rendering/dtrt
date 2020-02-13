class AreaLight:
    def __init__(self, shape_id, intensity, two_sided = False):
        self.type = 'area_light'
        self.shape_id = shape_id
        self.intensity = intensity
        self.two_sided = two_sided

class AreaLightEx:
    def __init__(self, shape_id, intensity, kappa):
        self.type = 'area_lightEx'
        self.shape_id = shape_id
        self.intensity = intensity
        self.kappa = kappa    
