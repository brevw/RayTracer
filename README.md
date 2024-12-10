## Ray Tracer
# Render Exmaple
```python
python main.py -i scenes/FinalRender.json
```
![Alt text](out/FinalRender.png)

# requirements 
```python
pip install -r requirements.txt
```
# Usage
commands are run through CLI and scenes are described with a json file that will be parsed later on into objects.

```
usage: main.py [-h] -i INFILE [INFILE ...] [-o OUTDIR] [-s] [-f FACTOR]

options:
  -h, --help            show this help message and exit
  -i INFILE [INFILE ...], --infile INFILE [INFILE ...]
                        Name of json file that will define the scene
  -o OUTDIR, --outdir OUTDIR
                        directory for output files
  -s, --show            Show the final image in a window
  -f FACTOR, --factor FACTOR
                        Scale factor for resolution
```


# Features implemented
- Mirror Reflections
- Refractions
- Motion Blur
- Depth of field 
- Area Light
- Quadrics
- MetaBalls
- Textures
- Phong-Shading