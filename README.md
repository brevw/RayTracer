# 261237854 - Ahmed Tlili

Extra things implemented with the format: extension (respective file)

- Mirror Reflections: (Reflection.png, AreaLightReflection.png)
    > "reflection_intensity" is added to each material.
- Refractions: (Refraction.png)
    > "refractive_index", "refraction_intensity" are added to each material.
- Motion Blur (MotionBlur.png)
    > Each scene now has a start and end time and geometry can move through start translation and end translation with start time of the movement and end time. "start_translation", "end_translation", "time_start", "time_end" for the geormetry were added.
- Depth of field (DOF.png)
    > every scene will now have "aperture_size" and "focal_distance" params.
- Area Light (AreaLight.png)
    > for every intersection compute lighting by randomly sampling over the surface.
    "emissive_color", "power", "attenuation" were added to every material object for emissive objects and "samples" for geometry indicating how much to sample over the surface.
- Quadrics (Quadrics.png)
    > new geometry class created.
- MetaBalls (MetaBall.png)
    > use of path tracing with numerical. New class created.
- Textures (Texture.json)
    > Texture class created in the helper file that is used to sample color with u and v scale. Now texture can reference a texture.
- Phong-Shading ()
    > can be activated by setting USE_PHONG_SHADING to true in the geometry file 



# Novel Scene and Competition Image
261237854-AhmedTlili-competition.json tried to use most of the features implemented