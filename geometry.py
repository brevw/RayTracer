import helperclasses as hc
import glm
import igl
import random
import math

USE_PHONG_SHADING = False

class Geometry:
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], samples: int):
        self.name = name
        self.gtype = gtype
        self.materials = materials
        self.emits_light = False
        self.samples = samples
        if materials:
            for mat in materials:
                if glm.length(mat.emissive_color) != 0:
                    self.emits_light = True
                    break

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        return intersect
    
    def sample_light(self) -> list[hc.Light]:
        """
        Base implementation of light sampling. Should be overridden by derived classes.
        Returns a list of (point, normal) pairs.
        """
        return []

class Sphere(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], center: glm.vec3, radius: float, samples: int):
        super().__init__(name, gtype, materials, samples)
        self.center = center
        self.radius = radius

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        p: glm.vec3 = ray.origin
        d: glm.vec3 = ray.direction
        l: glm.vec3 = ray.origin - self.center
        l_dot_l, l_dot_d = glm.dot(l, l), glm.dot(l, d)

        delta = l_dot_d * l_dot_d - (l_dot_l - self.radius * self.radius)
        if delta < 0:
            return
        delta = glm.sqrt(delta)
        t = -l_dot_d - delta
        t = t if t > 0 and t < intersect.t else -l_dot_d + delta
        if t > 0 and t < intersect.t:
            intersect.t = t 
            intersect.position = p + t * d
            intersect.normal = glm.normalize(intersect.position - self.center)
            intersect.mat = self.materials[0] if self.materials else None 
    def sample_light(self) -> list[hc.Light]:
        if not self.emits_light:
            return []
        light_instances = []
        material = self.materials[0]

        for _ in range(self.samples):
            # Generate a random point on the surface of the sphere
            theta = random.uniform(0, 2 * math.pi) 
            phi = random.uniform(0, math.pi)  
            x = self.center.x + self.radius * math.sin(phi) * math.cos(theta)
            y = self.center.y + self.radius * math.sin(phi) * math.sin(theta)
            z = self.center.z + self.radius * math.cos(phi)
            position = glm.vec3(x, y, z)

            # Normal at the point on the sphere
            normal = glm.normalize(position - self.center)

            # Create a light instance
            light = hc.Light(
                l_type = "point",
                name = f"{self.name}_light",
                colour = material.emissive_color,
                vector = position + 1e-3 * normal,
                attenuation = material.attenuation
            )
            light_instances.append(light)

        return light_instances

class MetaBall(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], samples: int, centers: list[glm.vec3], threshold: float):
        super().__init__(name, gtype, materials, samples)
        self.centers = centers
        self.threshold = threshold
    
    def eval(self, p: glm.vec3) -> float:
        val = 0.0
        for c in self.centers:
            val += 1.0 / (glm.length2(p - c) + 1e-4)
        return val

    def sdf(self, p: glm.vec3) -> float:
        return (self.threshold - self.eval(p))
    
    def grad(self, p: glm.vec3) -> glm.vec3:
        gradient = glm.vec3(0.0, 0.0, 0.0)
        for c in self.centers:
            diff = p - c 
            gradient += 2 *  diff / (glm.length2(diff) + 1e-3) ** 2
        return gradient

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        # use ray marching 
        p = ray.origin
        total_distance = 0

        if self.eval(p) < 1e-3:
            p += 0.6 * ray.direction
            total_distance += 0.6

        for _ in range(100):
            sdf_value = self.sdf(p)
            p += sdf_value * ray.direction
            total_distance += sdf_value
            if abs(sdf_value) < 1e-3:  # If close enough to the surface
                if total_distance < 1e-2:
                    return
                intersect.t = glm.length(ray.origin - p)
                intersect.mat = self.materials[0] if self.materials else None
                intersect.position = p
                intersect.normal = glm.normalize(self.grad(p))
                return
            

class Plane(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], point: glm.vec3, normal: glm.vec3, samples: int):
        super().__init__(name, gtype, materials, samples)
        self.point = point
        self.normal = normal

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        p = ray.origin
        d = ray.direction
        n = self.normal

        d_dot_n = glm.dot(d, n)
        if abs(d_dot_n) < 1e-6:
            return
        t = glm.dot((self.point - p), n) / d_dot_n
        if t > 0 and t < intersect.t: 
            intersect.t = t
            intersect.normal = n
            intersect.position = p + t * d
            intersect.mat = None if not self.materials else \
                            self.materials[1] if len(self.materials) > 1 and (int(glm.floor(intersect.position.x)) + int(glm.floor(intersect.position.z))) & 1 == 1 else self.materials[0]

class AABB(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], minpos: glm.vec3, maxpos: glm.vec3, samples: int):
        # dimension holds information for length of each size of the box
        # material index: 0: front, 1: back, 2: right, 3: left, 4: top, 5: bottom
        super().__init__(name, gtype, materials, samples)
        self.minpos = minpos
        self.maxpos = maxpos

    def face_to_mat_index(self, face):
        return min(face, len(self.materials) - 1)

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        p = ray.origin
        d = ray.direction
        
        t_enter, t_exit = -float('inf'), float('inf')
        enter_normal, exit_normal = glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 0.0, 0.0)


        # along each axis right_order 
        for i in range(3): 
            min_ = self.minpos[i]
            max_ = self.maxpos[i]
            d_ = d[i]
            p_ = p[i]

            if abs(d_) < 1e-6: 
                if not (min_ <= p_ <= max_):
                    return
            else: 
                t_min = (min_ - p_) / d_
                t_max = (max_ - p_) / d_
                
                #reordering t_min and t_max
                sign = 1
                if t_min > t_max:
                    t_min, t_max = t_max, t_min
                    sign = -1
                
                # intersect intervals
                # equivalent to t_enter = max(t_min, t_enter) but keeps track of enter normals
                if t_min > t_enter:
                    t_enter = t_min
                    enter_normal = glm.vec3(0.0, 0.0, 0.0)
                    enter_normal[i] = - sign


                # equivalent to t_exit = min(t_max, t_exit) but keeps track of the exit normal
                if t_max < t_exit:
                    t_exit = t_max
                    exit_normal = glm.vec3(0.0, 0.0, 0.0)
                    exit_normal[i] = sign 

                if t_exit < t_enter:
                    return 

        t = t_enter if t_enter > 0 else t_exit
        normal = enter_normal if t_enter > 0 else exit_normal
        if t > 0 and t < intersect.t: 
            intersect.t = t
            intersect.position = p + t * d 
            intersect.normal = normal
            if not self.materials:
                intersect.mat
            else:
                face = 0 if normal[2] == 1 else (1 if normal[2] == -1 else (2 if normal[0] == 1 else (3 if normal[0] == -1 else (4 if normal[1] == 1 else 5))))
            intersect.mat = self.materials[self.face_to_mat_index(face)]
    def sample_light(self) -> list[tuple[glm.vec3,glm.vec3]]:
        if not self.emits_light: 
            return []

        light_instances = []

        # Define face sampling by index
        faces = {
            0: {"u": 0, "v": 1, "constant": self.maxpos.z, "normal": glm.vec3(0, 0, 1)},  # Front (+Z)
            1: {"u": 0, "v": 1, "constant": self.minpos.z, "normal": glm.vec3(0, 0, -1)},  # Back (-Z)
            2: {"u": 1, "v": 2, "constant": self.maxpos.x, "normal": glm.vec3(1, 0, 0)},  # Right (+X)
            3: {"u": 1, "v": 2, "constant": self.minpos.x, "normal": glm.vec3(-1, 0, 0)},  # Left (-X)
            4: {"u": 0, "v": 2, "constant": self.maxpos.y, "normal": glm.vec3(0, 1, 0)},  # Top (+Y)
            5: {"u": 0, "v": 2, "constant": self.minpos.y, "normal": glm.vec3(0, -1, 0)},  # Bottom (-Y)
        }

        for face_index, face_data in faces.items():
            material = self.materials[self.face_to_mat_index(face_index)]
            if glm.length(material.emissive_color) == 0:
                continue  # Skip non-emissive faces

            # Sample points on the face
            for _ in range(self.samples):
                u_coord = random.uniform(self.minpos[face_data["u"]], self.maxpos[face_data["u"]])
                v_coord = random.uniform(self.minpos[face_data["v"]], self.maxpos[face_data["v"]])

                # Create a point based on the face
                point = glm.vec3(0.0)
                point[face_data["u"]] = u_coord
                point[face_data["v"]] = v_coord

                # Assign the constant coordinate for the face
                if face_data["normal"].x != 0:
                    point.x = face_data["constant"]
                elif face_data["normal"].y != 0:
                    point.y = face_data["constant"]
                elif face_data["normal"].z != 0:
                    point.z = face_data["constant"]

                # Create a light instance
                light = hc.Light(
                    ltype = "point", 
                    name = f"{self.name}_light_{face_index}",
                    colour = material.emissive_color * material.power / self.samples,
                    vector = point + 1e-3 * face_data["normal"],
                    attenuation = material.attenuation
                )
                light_instances.append(light)
        return light_instances

class Mesh(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], translate: glm.vec3, scale: float,
                 filepath: str, samples: int):
        super().__init__(name, gtype, materials, samples)
        verts, _, norms, self.faces, _, _ = igl.read_obj(filepath)
        self.verts = []
        self.norms = []
        for v in verts:
            v_ = (glm.vec3(v[0], v[1], v[2]) + translate) * scale
            self.verts.append(v_)
        # if norms are given don't compute them again 
        if self.norms:
            for n in norms:
                self.norms.append(glm.vec3(n[0], n[1], n[2]))
        else:
            for f in self.faces: 
                a, b, c = self.verts[f[0]], self.verts[f[1]], self.verts[f[2]]
                self.norms.append(glm.normalize(glm.cross(b - a, c - a)))

        # compute vertex normals
        vertex_normals = [ glm.vec3(0.0, 0.0, 0.0) for _ in range(len(self.verts))]
        for j, f in enumerate(self.faces):
            a_i, b_i, c_i = f
            n = self.norms[j]
            vertex_normals[a_i] += n
            vertex_normals[b_i] += n
            vertex_normals[c_i] += n
        self.vertex_normals = [glm.normalize(v_n) for v_n in vertex_normals] 


    def intersect_triangle(self, face_i: int, ray: hc.Ray, intersect: hc.Intersection):
        face = self.faces[face_i]

        p = ray.origin
        d = ray.direction
        n = self.norms[face_i]
        a, b, c = self.verts[face[0]], self.verts[face[1]], self.verts[face[2]]
        d_dot_n = glm.dot(d, n)
        if abs(d_dot_n) < 1e-6: 
            return
        t = glm.dot(a - p, n) / d_dot_n
        intersec_p = p + t * d
        alpha, beta, gamma = 0, 0, 0
        if t <= 0:
           return 
        signed_A_c = glm.dot(glm.cross(b - a, intersec_p - a), n)
        if signed_A_c <= 0:
            return
        signed_A_a = glm.dot(glm.cross(c - b, intersec_p - b), n)
        if signed_A_a <= 0:
            return 
        signed_A_b = glm.dot(glm.cross(a - c, intersec_p - c), n)
        if signed_A_b <= 0:
            return
        if t < intersect.t:
            intersect.t = t
            intersect.position = intersec_p
            # use phong shading and interpolate normal across neighbouring vertices
            if USE_PHONG_SHADING:
                n_a, coeff_a = self.vertex_normals[face[0]], signed_A_a / (signed_A_a + signed_A_b + signed_A_c)
                n_b, coeff_b = self.vertex_normals[face[1]], signed_A_b / (signed_A_a + signed_A_b + signed_A_c)
                n_c, coeff_c = self.vertex_normals[face[2]], signed_A_c / (signed_A_a + signed_A_b + signed_A_c)
                intersect.normal = glm.normalize( coeff_a * n_a + coeff_b * n_b + coeff_c * n_c )
            else: 
                intersect.normal = n
            intersect.mat = self.materials[0] if self.materials else None

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        for i in range(len(self.faces)):
            self.intersect_triangle(i, ray, intersect)

class Node(Geometry):
    def __init__(self, name: str, gtype: str, M: glm.mat4, materials: list[hc.Material], samples: int):
        super().__init__(name, gtype, materials, samples)        
        self.children: list[Geometry] = []
        self.M = M
        self.Minv = glm.inverse(M)
        self.emits_light = True # alway true easier way to handle this because of the parser

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        transformed_ray = hc.Ray((self.Minv * glm.vec4(ray.origin, 1.0)).xyz,
                                 (self.Minv * glm.vec4(ray.direction, 0.0)).xyz)
        for child in self.children:
            old_t = intersect.t
            child.intersect(transformed_ray, intersect)
            
            # if we find interseciton transform normal
            if intersect.t != old_t:
                intersect.position = (self.M * glm.vec4(intersect.position, 1.0)).xyz
                intersect.normal = glm.normalize(glm.transpose(self.Minv) * glm.vec4(intersect.normal, 0.0)).xyz
                if not intersect.mat: 
                    intersect.mat = self.materials[0]
    def sample_light(self) -> list[hc.Light]:
        l = []
        for c in self.children:
            l += c.sample_light()
        for l_ in l:
            l_.vector = self.M * l_.vector
        return l