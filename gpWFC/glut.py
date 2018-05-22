from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.raw.GL.VERSION.GL_1_5 import glBufferData as rawGlBufferData
from OpenGL.GLUT import *
from .runners import BacktrackingRunner

def get_ctx():
	import pyopencl as cl
	platform = cl.get_platforms()[0]
	print('getting ctx')

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
			'''.format(w=w, h=h) + '''
			in vec2 screenPos;
			out vec4 fragColor;

			layout(binding=1) uniform sharedState {
				uint[WORLD_W * WORLD_H * 2] worldState;
			};

			vec2 worldSize = vec2(WORLD_W, WORLD_H);
			vec2 tileSize = 1 / worldSize;

			void main() {
				vec2 tilePos;
				vec2 tileSpace = modf(screenPos * worldSize, tilePos);
				uint index  = uint(tilePos.x * WORLD_H + tilePos.y) * 2u;
				uint bits0 = worldState[index+0u];
				uint bits1 = worldState[index+1u];
				fragColor = vec4(
					float(bits0 <<  0 & 0xFFu) / 255.0,
					float(bits0 <<  8 & 0xFFu) / 255.0,
					float(bits0 << 16 & 0xFFu) / 255.0,
					0.0
				);
			}
		''', GL_FRAGMENT_SHADER)
		self.shader = shaders.compileProgram(vert, frag)

		ubo = None
		def gl_buffer_allocator(size):
			global ubo
			print('genbufs')
			ubo = glGenBuffers(1)
			glBindBuffer(GL_UNIFORM_BUFFER, ubo)
			rawGlBufferData(GL_ARRAY_BUFFER, size, None, GL_STATIC_DRAW)
			# glBufferStorage(GL_UNIFORM_BUFFER, size, None, GL_MAP_READ_BIT | GL_MAP_WRITE_BIT)
			# glBufferStorage(GL_UNIFORM_BUFFER, size, None, GL_DYNAMIC_STORAGE_BIT)
			glBindBuffer(GL_UNIFORM_BUFFER, 0)
			print('bind buf')
			return GLBuffer(self.ctx, mem_flags.READ_WRITE, int(ubo))

		ctx = get_ctx()
		print('got ctx')
		self.runner = BacktrackingRunner(model, Observer=CLObserver, Propagator=Propagator, ctx=ctx, allocator=gl_buffer_allocator)
		print('got ubo', ubo)
		# self.uniform_locations = dict((name, glGetUniformLocation(self.shader, name)) for name in ['worldState'])
		# blockIndex = glGetUniformBlockIndex(self.shader, 'worldState')
		# glUniformBlockBinding(self.shader, blockIndex, 1);
		glBindBufferBase(GL_UNIFORM_BUFFER, 1, ubo)
		print('bound buffer base')

	def launch(self):
		glutMainLoop()

	def display(self):
		glClear(GL_COLOR_BUFFER_BIT)
		shaders.glUseProgram(self.shader)
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		glFlush()

	def reshape(self, w, h):
		glViewport(0, 0, w, h)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glMatrixMode(GL_MODELVIEW)
