from pyopencl import create_some_context, GLBuffer, mem_flags, CommandQueue
from pyopencl.array import Array, to_device
import pyopencl as cl
from .observers import CLObserver
from .propagators import CL1Propagator

class Runner(object):
	def __init__(self, model, Observer=CLObserver, Propagator=CL1Propagator, ctx=None, allocator=None):
		if not ctx:
			ctx = create_some_context()
		self.ctx = ctx
		self.model = model

		self.observer = Observer(model, ctx=ctx)
		print('ob')
		self.propagator = Propagator(model, ctx=ctx)
		print('prop')

		self.grid_array = self.model.build_grid()
		self.snapshot = self.grid_array.copy()
		# with CommandQueue(ctx) as queue:
		queue = CommandQueue(ctx)
		if True:
			# self.grid = to_device(queue, self.grid_array, allocator=allocator)

			self.grid = Array(queue, self.grid_array.shape, self.grid_array.dtype, allocator=allocator)
			self.grid.queue = None
			self.grid.allocator = None
			# data = allocator(self.grid_array.nbytes)
			# self.grid = Array(ctx, self.grid_array.shape, self.grid_array.dtype, data=data)

			cl.enqueue_acquire_gl_objects(queue, [self.grid.base_data])

			self.grid.set(self.grid_array, queue=queue)
			self.candidate = self.observer.observe(self.grid, queue=queue)[1:]
			print('cand')

			cl.enqueue_release_gl_objects(queue, [self.grid.base_data])

		self.queue = queue
		self.done = False
		print('starting')

	def step(self):
		index, collapsed = self.candidate
		self.propagator.propagate(self.grid, index, collapsed)
		status = self.observer.observe(self.grid)

		if status[0] == 'continue':
			self.candidate = status[1:]
		elif status[0] == 'error':
			self.done = True
		else:
			self.grid.get(ary=self.grid_array)

		return status[0]

	def run(self):
		while not self.done:
			yield self.step()

	def finish(self):
		status = None
		for status in self.run():
			pass
		return status

class BacktrackingRunner(Runner):
	def __init__(self, *args, snapshot_every=4, **kwargs):
		super().__init__(*args, **kwargs)
		self.snapshot = self.grid_array.copy()
		self.snapshot_every = snapshot_every
		self.snapshot_age = 0

	def step(self):
		# with CommandQueue(self.ctx) as queue:
		queue = self.queue
		if True:
			cl.enqueue_acquire_gl_objects(queue, [self.grid.base_data])
			print('acq')

			index, collapsed = self.candidate

			# self.grid_array.ravel()[index] = collapsed

			# from pudb import set_trace; set_trace()
			# grid = self.grid.with_queue(queue)
			# grid.set(self.grid_array)

			# OR

			# self.grid.set(self.grid_array, queue=queue)


			self.propagator.propagate(self.grid, index, collapsed, queue=queue)
			print('propd')
			status = ('asd') # self.observer.observe(grid)
			print('obsd')
			# self.grid.get(ary=self.grid_array)

			if status[0] == 'continue':
				self.candidate = status[1:]
				self.snapshot_age += 1
				if self.snapshot_age >= self.snapshot_every:
					self.grid.get(ary=self.snapshot, queue=queue)
					self.snapshot_age = 0
			elif status[0] == 'error':
				if not self.snapshot is None:
					print('backtracking {} rounds'.format(self.snapshot_age))
					self.grid.set(self.snapshot, queue=queue)
					self.candidate = self.observer.observe(self.grid, queue=queue)[1:]
					self.snapshot = None
					self.snapshot_age = self.snapshot_every
					return self.step()
				else:
					print('cannot backtrack anymore')
					self.done = True
			else:
				self.done = True

			cl.enqueue_release_gl_objects(queue, [self.grid.base_data])
			print('rlsd')
		print('queue done')
		# return status[0]
