import glm
from PIL import Image

class Texture:
    def __init__(self, image_path) -> None:
        self.image = Image.open(image_path)
        self.width, self.height = self.image.size
        self.pixels = self.image.load()
    
    def sample(self, u, v):
        x = int(u * self.width) % self.width
        y = int(v * self.height) % self.height
        return glm.vec3(*[c / 255.0 for c in self.pixels[x, y][:3]])
    

class Ray:
    def __init__(self, o: glm.vec3, d: glm.vec3):
        self.origin = o
        self.direction = d

    def getDistance(self, point: glm.vec3):
        return glm.length(point - self.origin)

    def getPoint(self, t: float):
        return self.origin + self.direction * t

class Material:
    def __init__(self, name: str, diffuse: glm.vec3, specular: glm.vec3, shininess: float, reflection_intensity: float, emissive_color: glm.vec3, power: int, attenuation: list, refractive_index: float, refraction_intensity: float, texture_map: Texture):
        self.name = name
        self.diffuse = diffuse      # kd diffuse coefficient
        self.specular = specular    # ks specular coefficient
        self.shininess = shininess  # specular exponent            
        self.reflection_intensity = reflection_intensity # amount of light reflected
        self.emissive_color = emissive_color # for light surfaces (glm.vec3(0.0, 0.0, 0.0) for no light emitted)
        self.power = power
        self.attenuation = attenuation 
        self.refractive_index = refractive_index # use for refraction
        self.refraction_intensity = refraction_intensity # amount of light refracted
        self.texture = texture_map

class Light:
    def __init__(self, ltype: str, name: str, colour: glm.vec3, vector: glm.vec3, attenuation: glm.vec3):
        self.name = name
        self.type = ltype       # type is either "point" or "directional"
        self.colour = colour    # colour and intensity of the light
        self.vector = vector    # position, or normalized direction towards light, depending on the light type
        self.attenuation = attenuation # attenuation coeffs [quadratic, linear, constant] for point lights

class Intersection:
    def __init__(self, t: float, normal: glm.vec3, position: glm.vec3, material: Material, geom):
        self.t = t
        self.normal = normal
        self.position = position
        self.mat = material
        self.geom = geom

    @staticmethod
    def default(): # create an empty intersection record with t = inf
        t = float("inf")
        normal = None 
        position = None 
        mat = None 
        geom = None
        return Intersection(t, normal, position, mat, geom)
