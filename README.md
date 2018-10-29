gpu-collapse
============

Implementation of the [Wave Function Collapse][WFC] procedural content generation algorithm,
using [(py)OpenCL][pyopencl] for GPU acceleration.

Getting Started
---------------

make sure you have the python packages pyopencl, numpy and pyglet installed.

You can then run a basic example using

    python main.py

in the preview window the following keybindings are set:

- `escape`: close
- `space`: do one oberservation/propagation cycle and render
- `r`: cycle until stable, then render again
- `d`: debug view (overlay decimal display of bitmask for each tile)

There is also a more interesting sprite-based example that you can run using

    python circuit.py [render]

but as you can see I didn't set up the model constraints properly. Maybe you want to fix that?

`main.py` can take a few options that are just passed as strings on the command line, in any order.
They might not all be compatible with each other, in any case `main.py` is only a
starting point to write your own set up code with a more serious model.

### `cpu`

propagate using a simplistic CPU algorithm.

### `3d`

work in a 3d space (4x4x2 by default), with a *very* rudimentary preview.
more of a proof of concept, but totally workable.

In the 3d preview, the up and down keys can be used to cycle through slices of the Z axis.

### `silent`

don't open a preview or render, just measure the execution time.

### `render`

automatically step execution forward and take save a screenshot to `shots/0001.png` etc.
You can use e.g. ffmpeg to turn the png frames into an animation.

Programatic Usage
-----------------

`gpWFC` is set up to follow a 'mix and match' modular architecture as best as possible.
It is therefore divided into a couple of components that need to be used to run a simulation:

- the Tiles (`Tile` and `SpriteTile` from `models.py`):
  - `tile.weight` (float): the relative probability of occurence
  - `tile.compatible(other, direction_id)` (bool): constraint information
  - additional information for the Preview, e.g. `tile.image` and `tile.rotation` for `SpriteTile`
- the Model (`Model2d` and `Model3d` from `models.py`):
  - information about the *world*:
    - `model.world_shape` (tuple): dimensions of the world (any nr of axes)
    - `model.get_neighbours(pos)` (generator): tile adjacency information
  - information about the *tiles*:
    - `model.tiles` (list): the tiles to be used
    - `model.get_allowed_tiles(bitmask)` (list): a way to resolve the opaque bitmask
- the Runner (`Runner` and `BacktrackingRunner` from `runners.py`):
  - `runner.step()` (string): execute a single observartion/propagation cycle
  - `runner.finish()` (string): run the simulation until it either fails or stabilizes
  - `runner.run()` (generator): iterate over `runner.step()`
  - all of these return/yield status strings, which are one of:
    - `'done'` - fully collapsed
    - `'error'` - overconstrained / stuck
    - `'continue'` - step successful but uncollapsed tiles remain
- the Preview (`PreviewWindow*` from `previews.py`):
  - `preview.draw_tiles(pos, bits)`: draw the tiles at `pos` (tuple)
  - `preview.launch()`: enter interactive preview mode
  - `preview.render()`: enter non-interactive render loop
- the Observer and Propagator (`observers.py` and `propagators.py`):
  - you probably don't need to touch these

You can find a straightforward example of the basic setup steps in `circuit.py`, it should follow this flow:

- instantiate a Model
- instantiate Tiles and register them with the Model
- instantiate a Runner and pass it the Model
- instantiate a Preview and pass it the Runner
- launch the Preview

[WFC]: https://github.com/mxgmn/WaveFunctionCollapse
[pyopencl]: https://documen.tician.de/pyopencl
