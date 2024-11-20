import glm

class Ray:
    def __init__(self, o: glm.vec3, d: glm.vec3):
        self.origin = o
        self.direction = d

    def getDistance(self, point: glm.vec3):
        return glm.length(point - self.origin)

    def getPoint(self, t: float):
        return self.origin + self.direction * t

class Material:
    def __init__(self, name: str, diffuse: glm.vec3, specular: glm.vec3, shininess: float, reflection_intensity: float, emissive_color: glm.vec3, power: int):
        self.name = name
        self.diffuse = diffuse      # kd diffuse coefficient
        self.specular = specular    # ks specular coefficient
        self.shininess = shininess  # specular exponent            
        self.reflection_intensity = reflection_intensity # amount of light reflected
        self.emissive_color = emissive_color # for light surfaces (glm.vec3(0.0, 0.0, 0.0) for no light emitted)
        self.power = power

class Light:
    def __init__(self, ltype: str, name: str, colour: glm.vec3, vector: glm.vec3, attenuation: glm.vec3):
        self.name = name
        self.type = ltype       # type is either "point" or "directional"
        self.colour = colour    # colour and intensity of the light
        self.vector = vector    # position, or normalized direction towards light, depending on the light type
        self.attenuation = attenuation # attenuation coeffs [quadratic, linear, constant] for point lights

class Intersection:
    def __init__(self, t: float, normal: glm.vec3, position: glm.vec3, material: Material):
        self.t = t
        self.normal = normal
        self.position = position
        self.mat = material

    @staticmethod
    def default(): # create an empty intersection record with t = inf
        t = float("inf")
        normal = None 
        position = None 
        mat = None 
        return Intersection(t, normal, position, mat)
