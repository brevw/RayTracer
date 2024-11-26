import json
import helperclasses as hc
import geometry as geom
import scene
import glm

def make_vec3(array: list):
    return glm.vec3(array[0], array[1], array[2])

def make_matrix( t: glm.vec3, r: glm.vec3, s: glm.vec3 ):
    M = glm.mat4(1.0)
    M = glm.translate( M, t )
    M = glm.rotate( M, glm.radians(r.x), glm.vec3(1, 0, 0) )
    M = glm.rotate( M, glm.radians(r.y), glm.vec3(0, 1, 0) )
    M = glm.rotate( M, glm.radians(r.z), glm.vec3(0, 0, 1) )
    M = glm.scale( M, s )     
    return M   

def load_scene(infile: str):
    print("Parsing file:", infile)
    f = open(infile)
    data = json.load(f)

    # Loading camera
    cam_pos = make_vec3(data["camera"]["position"])
    cam_lookat = make_vec3(data["camera"]["lookAt"])
    cam_up = make_vec3(data["camera"]["up"])
    cam_fov = data["camera"]["fov"]

    # Loading resolution
    default_resolution = [1280, 720]    
    width = data.get("resolution", default_resolution)[0]
    height = data.get("resolution", default_resolution)[1]
        
    # Loading ambient light
    ambient = make_vec3(data.get("ambient", [0.1, 0.1, 0.1])) # set a reasonable default ambient light

    # Loading Anti-Aliasing options    
    jitter = data.get( "AA_jitter", False ) # default to no jitter
    samples = data.get( "AA_samples", 1 ) # default to no supersampling

    # DOF options
    aperture_size = data.get("aperture_size", 0.0) # defaults to no DOF
    focal_distance = data.get("focal_distance", 2.0) # defaults to two

    # max depth for reflection 
    depth = data.get("depth", 3) # max depth by default is 3
    
    # Loading scene lights
    lights = []    
    for light in data.get("lights", []):
        l_type = light["type"]
        l_name = light["name"]
        l_colour = make_vec3(light["colour"])            
        l_power = light.get( "power", 1.0 ) # The power scales the specified light colour
        if l_type == "point":
            l_vector = make_vec3(light["position"])
            l_attenuation = glm.vec3(1,0,0) if "attenuation" not in light else make_vec3( light["attenuation"] )
        elif l_type == "directional":
            l_vector = glm.normalize( make_vec3(light["direction"]) )
            l_attenuation = glm.vec3(0,0,0)
            if "attenuation" in light:
                print("Directional light", l_name, "has attenuation, ignoring")
        else:
            print("Unkown light type", l_type, ", skipping initialization")
            continue
        lights.append(hc.Light(l_type, l_name, l_colour * l_power, l_vector, l_attenuation))

    # Loading materials
    material_by_name = {} # materials dictionary 
    for material in data["materials"]:
        mat_name = material["name"]
        mat_diffuse = make_vec3(material["diffuse"])
        mat_specular = make_vec3(material["specular"])
        mat_shininess = 0 if "shininess" not in material else material["shininess"]
        reflection_intensity = 0 if "reflection_intensity" not in material else material["reflection_intensity"]
        emissive_color = glm.vec3(0.0, 0.0, 0.0) if "emissive_color" not in material else make_vec3(material["emissive_color"])
        attenuation = glm.vec3(1.0, 0.0, 0.0) if "attenuation" not in material else make_vec3(material["attenuation"])
        power = 1 if "power" not in material else material["power"]
        material_by_name[mat_name] = hc.Material(mat_name, mat_diffuse, mat_specular, mat_shininess, reflection_intensity, emissive_color, power, attenuation)


    # load geometires
    objects = [] # list of loaded object geometries and hierarchy roots
    geometry_by_name = {} # dictionary of geometries by name (for instances)

    for geometry in data["objects"]:
        g = load_geometry( geometry, material_by_name, geometry_by_name )
        objects.append(g)
        geometry_by_name[g.name] = g

    return scene.Scene(width, height, jitter, samples,  # General settings
                cam_pos, cam_lookat, cam_up, cam_fov,  # Camera settings
                ambient, lights,  # Light settings
                objects, # Geometries to render
                aperture_size, focal_distance, # DOF
                depth # for Reflections
                )

def load_geometry( geometry, material_by_name, geometry_by_name ):

    # Elements common to all objects: name, type, and material(s)
    g_name = geometry["name"]
    g_type = geometry["type"]
    g_mats = [ material_by_name[mat] for mat in geometry.get("materials",[]) ]
    g_samples = geometry.get("samples", 0)

    if g_type == "sphere":
        g_pos = make_vec3(geometry.get("position", [0, 0, 0]))
        g_radius = geometry["radius"]
        return geom.Sphere(g_name, g_type, g_mats, g_pos, g_radius, g_samples)
    elif g_type == "quadric":
        matrix_list = geometry.get("Q")
        flattened = [
            matrix_list[0][0], matrix_list[1][0], matrix_list[2][0], matrix_list[3][0],
            matrix_list[0][1], matrix_list[1][1], matrix_list[2][1], matrix_list[3][1],
            matrix_list[0][2], matrix_list[1][2], matrix_list[2][2], matrix_list[3][2],
            matrix_list[0][3], matrix_list[1][3], matrix_list[2][3], matrix_list[3][3]
        ]
        g_Q = glm.mat4(*flattened)
        inf = 1000
        g_bounds_min = geometry.get("bounds_min", [-inf, -inf, -inf])
        g_bounds_max = geometry.get("bounds_max", [inf, inf, inf])
        return geom.Quadrics(g_name, g_type, g_mats, g_samples, g_Q, g_bounds_min, g_bounds_max)
    elif g_type == "metaball":
        g_centers = [make_vec3(x) for x in geometry["centers"]]
        g_threshold = geometry["threshold"]
        return geom.MetaBall(g_name, g_type, g_mats, g_samples, g_centers, g_threshold)
    elif g_type == "plane":
        g_pos = make_vec3(geometry.get("position", [0, 0, 0]))
        g_normal = make_vec3(geometry["normal"])
        return geom.Plane(g_name, g_type, g_mats, g_pos, g_normal, g_samples)
    elif g_type == "box":
        minpos = make_vec3(geometry.get("min",[-1,-1,-1]))
        maxpos = make_vec3(geometry.get("max",[1,1,1]))
        return geom.AABB(g_name, g_type, g_mats, minpos, maxpos, g_samples)
    elif g_type == "mesh":
        g_path = geometry["filepath"]
        g_pos = make_vec3(geometry.get("position", [0, 0, 0]))
        g_scale = geometry["scale"]
        return geom.Mesh(g_name, g_type, g_mats, g_pos, g_scale, g_path, g_samples)
    elif g_type == "instance":
        g_pos = make_vec3(geometry.get("position", [0, 0, 0]))
        g_r = make_vec3(geometry.get("rotation", [0, 0, 0]))
        g_s = make_vec3(geometry.get("scale", [1, 1, 1]))
        M = make_matrix(g_pos, g_r, g_s)
        node = geom.Node(g_name, g_type, M, g_mats, g_samples)
        node.children.append( geometry_by_name[geometry["ref"]] )
        return node
    elif g_type == "node":
        g_pos = make_vec3(geometry.get("position", [0, 0, 0]))
        g_r = make_vec3(geometry.get("rotation", [0, 0, 0]))
        g_s = make_vec3(geometry.get("scale", [1, 1, 1]))
        M = make_matrix(g_pos, g_r, g_s)
        node = geom.Node(g_name, g_type, M, g_mats, g_samples)
        for child in geometry["children"]:
            g = load_geometry(child, material_by_name, geometry_by_name)
            node.children.append(g)
            geometry_by_name[g.name] = g
        return node
    else:
        print("Unkown object type", g_type, ", skipping initialization")
        return None