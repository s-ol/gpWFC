from numpy import ndenumerate
from pyglet.app import run
from pyglet.window import Window, key
from pyglet.resource import image
from pyglet.image import get_buffer_manager
from pyglet.text import Label
from pyglet.sprite import Sprite

class BasePreview(Window):
	def __init__(self, runner, width=512, height=512):
		super().__init__(width=width, height=height)
		self.runner = runner
		self.debug = False

	def on_draw(self):
		self.clear()

		for pos, bits in ndenumerate(self.runner.grid_array):
			self.draw_tiles(pos, bits)

	def on_key_press(self, symbol, modifiers):
		if symbol == key.ESCAPE:
			self.close()
		elif symbol == key.SPACE:
			self.runner.step()
		elif symbol == key.R:
			self.runner.finish()
		elif symbol == key.D:
			self.debug = not self.debug

	def screenshot(self, name='shots/snapshot.png'):
		get_buffer_manager().get_color_buffer().save(name)

	def render(self):
		iteration = 0
		self.dispatch_event('on_draw')
		self.flip()
		self.screenshot('shots/{:04}.png'.format(iteration))
		for i in runner.run():
			self.dispatch_events()
			self.dispatch_event('on_draw')
			self.flip()
			self.screenshot('shots/{:04}.png'.format(iteration))
			iteration += 1

	def launch(self):
		run()

class PreviewWindow(BasePreview):
	colors = ( (0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255) )
	rotations = [0, 90, 180, 270]
	def __init__(self, runner):
		super().__init__(runner, width=512, height=512)

		tile = image('tile.png')
		tile.anchor_x = 32
		tile.anchor_y = 32
		self.sprite = Sprite(img=tile, x=0, y=0)

	def draw_tiles(self, pos, bits):
		if bits == 0:
			return

		x, y = pos[-2:]
		self.sprite.x = x * 64 + 32
		self.sprite.y = self.height - y * 64 - 32

		tiles = self.runner.model.get_allowed_tiles(bits)
		self.sprite.opacity = 255 / len(tiles)

		for tile in tiles:
			for direction, adj in enumerate(tile.adj):
				if adj < 1 or self.rotations[direction] == None:
					continue
				self.sprite.color = self.colors[adj]
				self.sprite.rotation = self.rotations[direction] 
				self.sprite.draw()

		if self.debug:
			Label(str(bits), x=self.sprite.x, y=self.sprite.y).draw()

class PreviewWindow3d(PreviewWindow):
	rotations = [0, 90, None, 180, 270, None]

	def __init__(self, *args):
		super().__init__(*args)
		self.slice = 0

	def on_draw(self):
		self.clear()

		for pos, bits in ndenumerate(self.runner.grid_array[...,self.slice]):
			self.draw_tiles(pos, bits)

	def on_key_press(self, symbol, modifiers):
		if symbol == key.UP:
			self.slice += 1
		elif symbol == key.DOWN:
			self.slice += 1
		else:
			super().on_key_press(symbol, modifiers)
			return
		self.slice = self.slice % self.runner.model.world_shape[-1]
		print(self.slice)

class SpritePreviewWindow(BasePreview):
	def __init__(self, runner, tile_size):
		width = runner.model.world_shape[0] * tile_size
		height = runner.model.world_shape[1] * tile_size
		super().__init__(runner, width=width, height=height)

		self.sprite = Sprite(img=runner.model.tiles[0].image)
		self.tile_size = tile_size

	def draw_tiles(self, pos, bits):
		if bits == 0:
			return

		x, y = pos[-2:]
		self.sprite.x = x * self.tile_size + self.tile_size/2
		self.sprite.y = self.height - y * self.tile_size - self.tile_size/2

		tiles = self.runner.model.get_allowed_tiles(bits)
		self.sprite.opacity = 255 / len(tiles)

		for tile in tiles:
			tile.image.anchor_x = self.tile_size/2
			tile.image.anchor_y = self.tile_size/2
			self.sprite.image = tile.image
			self.sprite.rotation = tile.rotation * 90
			self.sprite.draw()

		if self.debug:
			Label(str(bits), x=self.sprite.x, y=self.sprite.y).draw()
