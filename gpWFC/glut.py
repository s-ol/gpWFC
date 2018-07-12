from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.raw.GL.VERSION.GL_1_5 import glBufferData as rawGlBufferData
from OpenGL.GLUT import *
from .runners import BacktrackingRunner
from pyopencl import GLBuffer, mem_flags

def get_ctx():
	import pyopencl as cl
	platform = cl.get_platforms()[0]

	from pyopencl.tools import get_gl_sharing_context_properties
	import sys
	if sys.platform == "darwin":
		return cl.Context(properties=get_gl_sharing_context_properties(),
				devices=[])
	else:
		# Some OSs prefer clCreateContextFromType, some prefer
		# clCreateContext. Try both.
		try:
			return cl.Context(properties=[
				(cl.context_properties.PLATFORM, platform)]
				+ get_gl_sharing_context_properties())
		except:
			return cl.Context(properties=[
				(cl.context_properties.PLATFORM, platform)]
				+ get_gl_sharing_context_properties(),
				devices = [platform.get_devices()[0]])

class GLUTWindow(object):
	def __init__(self, model):
		glutInit()
		glutInitWindowSize(512, 512)
		# glutInitWindowPosition(0, 0)
		glutCreateWindow('gpWFC')
		glutDisplayFunc(self.display)
		glutReshapeFunc(self.reshape)
		glutKeyboardFunc(self.keydown)

		vert = shaders.compileShader('''
			#version 420 core
			const vec2 quadVertices[4] = vec2[](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0) );

			out vec4 gl_Position;
			out vec2 screenPos;

			void main() {
				gl_Position = vec4(quadVertices[gl_VertexID], 0.0, 1.0);
				screenPos = gl_Position.xy / 2.0 + 0.5;
				screenPos.y = 1 - screenPos.y;
			}
		''', GL_VERTEX_SHADER)

		w, h = model.world_shape
		frag = shaders.compileShader(
			'''
			#version 420 core
			#define WORLD_W {w}
			#define WORLD_H {h}
			#define BIT_GRID 8
			'''.format(w=w, h=h, tiles=len(model.tiles)) + '''
			in vec2 screenPos;
			out vec4 fragColor;

			layout(binding=1) uniform sharedState {
				uvec2[WORLD_W * WORLD_H] worldState;
			};

			const vec2 worldSize = vec2(WORLD_W, WORLD_H);
			const vec2 tileSize = 1 / worldSize;
			const vec2 borders = vec2(0.02);

			float grid(vec2 pos, float w) {
				vec2 dark = smoothstep(w/2, w, pos) * smoothstep(1-w/2, 1-w, pos);
				return min(dark.x, dark.y);
			}
			float grid(vec2 pos) { return grid(pos, 0.02); }

			float bitgrid(vec2 tilepos, uvec2 bits, uint tiles) {
				uint bit = uint(dot(uvec2(tilepos * BIT_GRID), uvec2(1u, BIT_GRID)));
				if (bit >= tiles) return 1.0;
				return grid(mod(tilepos * 8.0, 1.0), 0.1) *
					(min(1, float(bits.x & (1u << bit))) * 0.5 + 0.25);
			}

			void main() {
				vec2 tilePos;
				vec2 tileSpace = modf(screenPos * worldSize, tilePos);
				uint index  = uint(tilePos.x * WORLD_H + tilePos.y);
				uvec2 bits = worldState[index].xy;

				fragColor = vec4(grid(tileSpace));
				fragColor *= bitgrid(tileSpace, bits, 32u);

				/*
				fragColor *= vec4(
					float(bits.x >>  0 & 0xFFu) / 255.0,
					float(bits.x >>  8 & 0xFFu) / 255.0,
					float(bits.x >> 16 & 0xFFu) / 255.0,
					1
				);
				*/
			}
		''', GL_FRAGMENT_SHADER)
		self.shader = shaders.compileProgram(vert, frag)

		store = {}
		def gl_buffer_allocator(size):
			if 'ubo' in store:
				raise Exception('noo')
			#global ubo
			#if ubo:
			#	print("nooo")
			print('genbufs')
			ubo = glGenBuffers(1)
			glBindBuffer(GL_UNIFORM_BUFFER, ubo)
			# rawGlBufferData(GL_UNIFORM_BUFFER, size, None, GL_STATIC_DRAW)
			# glBufferStorage(GL_UNIFORM_BUFFER, size, None, GL_DYNAMIC_STORAGE_BIT)
			glBufferStorage(GL_UNIFORM_BUFFER, size, None, GL_MAP_READ_BIT | GL_MAP_WRITE_BIT)
			glBindBuffer(GL_UNIFORM_BUFFER, 0)
			store['ubo'] = ubo
			print('bind buf')
			return GLBuffer(ctx, mem_flags.READ_WRITE, int(ubo))

		ctx = get_ctx()
		print('got ctx')
		self.runner = BacktrackingRunner(model, ctx=ctx, allocator=gl_buffer_allocator)
		ubo = store['ubo']
		print('got ubo', ubo)
		# self.uniform_locations = dict((name, glGetUniformLocation(self.shader, name)) for name in ['worldState'])
		# blockIndex = glGetUniformBlockIndex(self.shader, 'worldState')
		# glUniformBlockBinding(self.shader, blockIndex, 1);
		glBindBufferBase(GL_UNIFORM_BUFFER, 1, ubo)
		print('bound buffer base')

	def launch(self):
		print('starting main')
		glutMainLoop()

	def display(self):
		print('display')
		glClear(GL_COLOR_BUFFER_BIT)
		shaders.glUseProgram(self.shader)
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		glFlush()

	def reshape(self, w, h):
		glViewport(0, 0, w, h)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glMatrixMode(GL_MODELVIEW)
	
	def keydown(self, k, x, y):
		if k == b' ':
			print('stop')
			self.runner.step()
			print('redisp')
			glutPostRedisplay()
			print('redosp')
		elif k == b'r':
			glutPostRedisplay()
		elif ord(k) == 27: # Escape
			sys.exit(0)
		print('dun')
