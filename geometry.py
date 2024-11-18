import helperclasses as hc
import glm
import igl

USE_PHONG_SHADING = False

class Geometry:
    def __init__(self, name: str, gtype: str, materials: list[hc.Material]):
        self.name = name
        self.gtype = gtype
        self.materials = materials

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        return intersect

class Sphere(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], center: glm.vec3, radius: float):
        super().__init__(name, gtype, materials)
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


class Plane(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], point: glm.vec3, normal: glm.vec3):
        super().__init__(name, gtype, materials)
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
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], minpos: glm.vec3, maxpos: glm.vec3):
        # dimension holds information for length of each size of the box
        super().__init__(name, gtype, materials)
        self.minpos = minpos
        self.maxpos = maxpos

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
            intersect.mat = self.materials[0] if self.materials else None 



class Mesh(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], translate: glm.vec3, scale: float,
                 filepath: str):
        super().__init__(name, gtype, materials)
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
    def __init__(self, name: str, gtype: str, M: glm.mat4, materials: list[hc.Material]):
        super().__init__(name, gtype, materials)        
        self.children: list[Geometry] = []
        self.M = M
        self.Minv = glm.inverse(M)

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