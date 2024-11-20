import math
import glm
import numpy as np
import geometry as geom
import helperclasses as hc
from tqdm import tqdm
import random

PERSPECTIVE = True
INF: float = float('+inf')
def NOTHING_COLOR() -> glm.vec3:
    return glm.vec3(0.0, 0.0, 0.0)

class Scene:

    def __init__(self,
                 width: int,
                 height: int,
                 jitter: bool,
                 samples: int,
                 eye_position: glm.vec3,
                 lookat: glm.vec3,
                 up: glm.vec3,
                 fov: float,
                 ambient: glm.vec3,
                 lights: list[hc.Light],
                 objects: list[geom.Geometry],
                 aperture_size: float,
                 focal_distance: float,
                 depth: int
                 ):
        self.width = width  # width of image
        self.height = height  # height of image
        self.aspect = width / height  # aspect ratio
        self.jitter = jitter  # should rays be jittered
        self.samples = samples  # number of rays per pixel
        self.eye_position = eye_position  # camera position in 3D
        self.lookat = lookat  # camera look at vector
        self.up = up  # camera up position
        self.fov = fov  # camera field of view
        self.ambient = ambient  # ambient lighting
        self.lights = lights  # all lights in the scene
        self.objects = objects  # all objects in the scene

        # params used for depth of field 
        self.aperture_size = aperture_size # to simulate lens 
        self.focal_distance = focal_distance  # Plane in which everything is in focus

        # for mirror reflection
        self.depth = depth

    def render(self):
        image = np.zeros((self.height, self.width, 3)) # image with row,col indices and 3 channels, origin is top left
        context = self.get_global_execution_context()

        for col in tqdm(range(self.width)):
            for row in range(self.height):
                # our coord system will consider (i, j) == (0, 0) in the bottom left corner
                # i sweeps col and j sweeps rows
                i, j = col, self.height - 1 - row
                pixel_color = self.compute_pixel_color(i, j, context)

                image[row, col] = np.clip(np.array(pixel_color), 0.0, 1.0)

        return image
    
    def compute_pixel_color(self, i: int, j: int, context: dict) -> glm.vec3:
        """
        Compute the color of a single pixel.
        Could cast multiple rays if we have supersampling.
        """
        color = NOTHING_COLOR()
        # supersampling loop
        for sub_pixel_i in range(self.samples): 
            for sub_pixel_j in range(self.samples):
                r = self.generate_ray(i, j, sub_pixel_i, sub_pixel_j, context)
                color += self.trace_ray(r, self.depth)
        return color / (self.samples * self.samples)

    def trace_ray(self, ray: hc.Ray, depth: int) -> glm.vec3:
        """
        Trace a ray through the scene and compute the color at the intersection.
        """
        # Base Case 
        if depth <= 0: 
            return NOTHING_COLOR()

        # Test for intersection with all objects
        intersection = hc.Intersection.default()
        for obj in self.objects:
            obj.intersect(ray, intersection)                   

        # Perform shading computations on the intersection point
        if intersection.t < INF:
            base_color = self.shade(intersection)
            if intersection.mat.reflection_intensity <= 0: 
                return base_color
            reflected_direction = 2 * glm.dot(intersection.normal, -ray.direction) * intersection.normal + ray.direction
            reflected_ray = hc.Ray(intersection.position + 1e-4 * intersection.normal, reflected_direction)

            # Compute reflected color 
            reflected_color = self.trace_ray(reflected_ray, depth - 1)
            return glm.mix(base_color, reflected_color, intersection.mat.reflection_intensity)

        return NOTHING_COLOR()

    def shade(self, intersection: hc.Intersection) -> glm.vec3:
        """
        Perform shading calculations for an intersection.
        """
        mat = intersection.mat
        color_sub = NOTHING_COLOR()
        if not mat:
            return color_sub
        
        # for objects emitting light 
        if mat.emissive_color != NOTHING_COLOR():
            return mat.emissive_color * mat.power
        
        # shading from light sources
        color_sub += self.ambient * mat.diffuse
        for light in self.lights:
            light_dir: glm.vec3 = glm.normalize(light.vector - intersection.position)
            eps = 1e-5
            
            # move eps through normal so that we don't have self shadowing 
            light_ray = hc.Ray(intersection.position + eps * intersection.normal, light_dir)
            # check is light ray is occluded
            dist_to_light = glm.length(light.vector - intersection.position)
                
            if not self.is_light_occluded(light_ray, dist_to_light):
                color_sub += self.compute_lighting(intersection, light, light_dir, dist_to_light)

        # shading from objects emitting light
        emitted_color = NOTHING_COLOR()
        for obj in self.objects:
            if not obj.emits_light:
                continue

            # Sample lights from the object's surface
            sampled_lights = obj.sample_light()
            for light in sampled_lights:
                light_dir = glm.normalize(light.vector - intersection.position)
                eps = 1e-5

                # Create a light ray to check visibility
                light_ray = hc.Ray(intersection.position + eps * intersection.normal, light_dir)
                dist_to_light = glm.length(light.vector - intersection.position)

                if not self.is_light_occluded(light_ray, dist_to_light):
                    emitted_color += self.compute_lighting(intersection, light, light_dir, dist_to_light)
        return color_sub + emitted_color

    def compute_lighting(self, intersection: hc.Intersection, light: hc.Light, light_dir: glm.vec3, dist_to_light: float) -> glm.vec3:
        """
        Compute diffuse and specular lighting contributions.
        Here we suppose that the light is not occluded with respect to the intersection point.
        """
        mat = intersection.mat
        l_ = light_dir

        n_: glm.vec3 = intersection.normal
        v_: glm.vec3 = glm.normalize(self.eye_position - intersection.position)
        h_: glm.vec3 = glm.normalize(v_ + l_)

        diffuse  = max(0.0, glm.dot(n_, l_)) * mat.diffuse
        specular = math.pow(max(0.0, glm.dot(n_, h_)), mat.shininess) * mat.specular

        color_light = (diffuse + specular) * light.colour
        attenuation = 1.0 / (light.attenuation[2] + light.attenuation[1] * dist_to_light + light.attenuation[0] * dist_to_light * dist_to_light)
        return color_light * attenuation
    
    def is_light_occluded(self, light_ray: hc.Ray, dist_to_light: float) -> bool:
        """
        Check if a light ray is occluded by any object.
        """
        intersection_light_ray = hc.Intersection.default()
        for obj in self.objects:
            obj.intersect(light_ray, intersection_light_ray)
            if intersection_light_ray.t < dist_to_light: 
                return True
        return False

    def get_global_execution_context(self) -> dict:
        """
        Calculate the camera parameters and other constants needed for the execution.
        """
        cam_dir = self.eye_position - self.lookat
        distance_to_plane = 1.0
        top = distance_to_plane * math.tan(0.5 * math.pi * self.fov / 180)
        right = self.aspect * top
        bottom = -top
        left = -right

        w = glm.normalize(cam_dir)
        u = glm.cross(self.up, w)
        u = glm.normalize(u)
        v = glm.cross(w, u)

        const_pixel_width = (right - left) / self.width
        const_pixel_height = (top - bottom) / self.height
        const_sub_offset_width = (right - left) / (self.width * self.samples)
        const_sub_offset_height = (top - bottom) / (self.height * self.samples)

        return {
            'w': w, 'u': u, 'v': v,
            'left': left, 'right': right,
            'top': top, 'bottom': bottom,
            'const_pixel_width': const_pixel_width,
            'const_pixel_height': const_pixel_height,
            'const_sub_offset_width': const_sub_offset_width,
            'const_sub_offset_height': const_sub_offset_height,
            'distance_to_plane': distance_to_plane
        }

    def generate_ray(self, i: int, j: int, sub_pixel_i: int, sub_pixel_j: int, context: dict, use_DOF: bool = False) -> hc.Ray: 
        """
        Generate a ray for a given pixel and subpixel indices.
        """
        # compute offset inside pixel space
        offset_x = (sub_pixel_i + 0.5) * context["const_sub_offset_width"]
        offset_y = (sub_pixel_j + 0.5) * context["const_sub_offset_height"]

        if self.jitter: 
            offset_x += random.uniform(-0.5, 0.5) * context["const_sub_offset_width"]
            offset_y += random.uniform(-0.5, 0.5) * context["const_sub_offset_height"]

        # Generate rays
        o_u_euvw: float = context["left"]   + context["const_pixel_width"] * i + offset_x
        o_v_euvw: float = context["bottom"] + context["const_pixel_height"] * j + offset_y
        o_w_euvw: float = - context["distance_to_plane"]
        ray_origin = o_u_euvw * context["u"] + o_v_euvw * context["v"] + o_w_euvw * context["w"] + self.eye_position
        
        # we have two possible directions if orthographic or perspective
        ray_direction = glm.normalize(ray_origin - self.eye_position) if PERSPECTIVE else -context["w"]

        # Applt DOF (Depth of Field) if aperature is non zero 
        if self.aperture_size > 0.0: 
            lens_u = random.uniform(-1, 1) * self.aperture_size
            lens_v = random.uniform(-1, 1) * self.aperture_size

            # Compute focus point the one that lies in the focus plane at fixed distance
            # this is because the ray  that passes through the middle is not bended
            w = context["w"]  # Camera's viewing direction
            denominator = -glm.dot(ray_direction, w)
            if abs(denominator) > 1e-6:
                t = self.focal_distance / denominator
                focus_point = self.eye_position + ray_direction * t
            else:
                focus_point = self.eye_position + ray_direction * self.focal_distance

            # ajust ray 
            ray_origin = ray_origin + lens_u * context["u"] + lens_v * context["v"]
            ray_direction = glm.normalize(focus_point - ray_origin)

        return hc.Ray(ray_origin, ray_direction)