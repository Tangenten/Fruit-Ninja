import glob
import math
import random

import pygame
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d

def pointInsideShape(point, poly, include_edges = True):
	x, y = point
	n = len(poly)
	inside = False

	p1x, p1y = poly[0]
	for i in range(1, n + 1):
		p2x, p2y = poly[i % n]
		if p1y == p2y:
			if y == p1y:
				if min(p1x, p2x) <= x <= max(p1x, p2x):
					inside = include_edges
					break
				elif x < min(p1x, p2x):
					inside = not inside
		else:
			if min(p1y, p2y) <= y <= max(p1y, p2y):
				xinters = (y - p1y) * (p2x - p1x) / float(p2y - p1y) + p1x
				if x == xinters:
					inside = include_edges
					break
				if x < xinters:
					inside = not inside

		p1x, p1y = p2x, p2y


	return inside


def lineNormals(line):
	dx = line[1][0] - line[0][0]
	dy = line[1][1] - line[0][1]

	return [[-dy, dx], [dy, -dx]]


def lineIntersection(line1, line2):
	def slope(p1, p2):
		try:
			return (p2[1] - p1[1]) / (p2[0] - p1[0])
		except ZeroDivisionError:
			return p2[1] - p1[1]

	def yIntercept(slope, p1):
		return p1[1] - 1. * slope * p1[0]

	m1 = slope(line1[0], line1[1])
	b1 = yIntercept(m1, line1[0])
	m2 = slope(line2[0], line2[1])
	b2 = yIntercept(m2, line2[0])

	# min_allowed = 1e-5  # guard against overflow
	# big_value = 1e10  # use instead (if overflow would have occurred)
	# if abs(m1 - m2) < min_allowed:
	# 	x = big_value
	# else:
	# 	x = (b2 - b1) / (m1 - m2)

	try:
		x = (b2 - b1) / (m1 - m2)
	except ZeroDivisionError:
		x = (b2 - b1)

	y = m1 * x + b1
	# y2 = m2 * x + b2

	return [x, y]


def lineSegmentIntersection(line1, line2):
	intersectionPoint = lineIntersection(line1, line2)

	if (min(line1[0][0], line1[1][0]) - 1) <= intersectionPoint[0] <= (max(line1[0][0], line1[1][0]) + 1):
		if (min(line1[0][1], line1[1][1]) - 1) <= intersectionPoint[1] <= (max(line1[0][1], line1[1][1]) + 1):
			if (min(line2[0][0], line2[1][0]) - 1) <= intersectionPoint[0] <= (max(line2[0][0], line2[1][0]) + 1):
				if (min(line2[0][1], line2[1][1]) - 1) <= intersectionPoint[1] <= (max(line2[0][1], line2[1][1]) + 1):
					return intersectionPoint

	return []

def distanceBetweenPoints(p1, p2):
	return math.sqrt( (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 )

def centerFromPoints(points):
	return [sum([p[0] for p in points]) / len(points), sum([p[1] for p in points]) / len(points)]

def angleBetweenPoints(p1, p2):
	return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

def angleToVector(angle):
	return [math.cos(angle), math.sin(angle)]


class game:

	def __init__(self):
		# ctypes.windll.user32.SetProcessDPIAware()
		pygame.mixer.pre_init(44100, 16, 2, 1024)
		pygame.init()
		pygame.display.set_caption("Fruit Ninja")
		# pygame.display.set_icon()

		pygame.key.set_repeat(200, 1)

		self.space = pymunk.Space(threaded = True)
		self.space.threads = 2
		self.space.iterations = 5
		self.space.gravity = 0, 1000
		self.stepSize = 0.005
		self.carryOver = 0

		self.running = False

		self.clock = pygame.time.Clock()
		self.deltaTime = 0.016
		self.FPS = 0
		self.targetFPS = 1000

		self.width = 1280
		self.height = 720
		self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
		self.virtualWidth = 1280 / 2
		self.virtualHeight = 720 / 2
		self.virtualScreen = pygame.Surface([self.virtualWidth, self.virtualHeight])
		self.widthPixelsPerVirtualPixel = self.width / self.virtualWidth
		self.heightPixelsPerVirtualPixel = self.height / self.virtualHeight

		self.clickPosition = [0.0, 0.0]
		self.releasePosition = [0.0, 0.0]
		self.currentPosition = [0.0, 0.0]
		self.previousPosition = [0.0, 0.0]
		self.holdingState = False

		self.shapes = Shapes(self.space, self.virtualWidth, self.virtualHeight)

		floorBody = pymunk.Body(body_type = pymunk.Body.STATIC)
		floorBody.position = [0, self.virtualHeight]
		floorLine1 = pymunk.Segment(floorBody, [0, 0], [self.virtualWidth, 0], 2)
		floorLine1.friction = 1

		self.space.add(floorLine1)

		self.a = pymunk.Body(10, 1666, pymunk.Body.DYNAMIC)
		self.a.position = [100, 100]
		self.ap = pymunk.Poly.create_box(self.a, [40, 40])
		self.c = None

		self.sliceSounds = []
		for sound_file in glob.glob("sounds/*.wav"):
			self.sliceSounds.append(pygame.mixer.Sound(sound_file))

		#self.options = pymunk.pygame_util.DrawOptions(self.virtualScreen)
		pymunk.pygame_util.positive_y_is_up = False

	def start(self):
		self.render()

	def stop(self):
		self.running = False

	def render(self):
		self.running = True
		while self.running:

			self.carryOver += self.deltaTime
			while self.carryOver > self.stepSize:
				self.space.step(self.stepSize)
				self.carryOver -= self.stepSize

			if self.shapes.didSlice:
				self.virtualScreen.fill((80, 80, 80))
				random_sound = random.choice(self.sliceSounds)
				random_sound.set_volume(0.2)
				random_sound.play()

			self.shapes.update(self.deltaTime)

			linesArr = self.shapes.getLines()
			for lines in linesArr:
				pygame.draw.lines(self.virtualScreen, [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)], True, lines, 1)

			for shape in self.shapes.shapes:
				p = shape.getPosition()
				pygame.draw.rect(self.virtualScreen, [255, 0, 0], [p[0], p[1], 4, 4])

		#	self.space.debug_draw(self.options)

			if self.holdingState:
				pygame.draw.line(self.virtualScreen, [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)], self.clickPosition, self.currentPosition)

			pygame.transform.scale(self.virtualScreen, [self.width, self.height], self.screen)
			pygame.display.update()

			self.clock.tick(self.targetFPS)
			self.deltaTime = self.clock.get_time() / 1000.0
			self.FPS = self.clock.get_fps()
			print(self.FPS)

			self.input()
			self.virtualScreen.fill((0, 0, 0))

	def input(self):
		events = pygame.event.get()
		for event in events:
			if event.type == pygame.QUIT:
				self.stop()
			if event.type == pygame.VIDEORESIZE:
				self.width = event.dict['size'][0]
				self.height = event.dict['size'][1]
				self.screen = pygame.display.set_mode((self.width, self.height), pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE)
				self.widthPixelsPerVirtualPixel = self.width / self.virtualWidth
				self.heightPixelsPerVirtualPixel = self.height / self.virtualHeight
			if event.type == pygame.MOUSEBUTTONDOWN:
				self.clickPosition = list(pygame.mouse.get_pos())
				self.clickPosition[0] = self.clickPosition[0] / self.widthPixelsPerVirtualPixel
				self.clickPosition[1] = self.clickPosition[1] / self.heightPixelsPerVirtualPixel
				self.holdingState = True
			elif event.type == pygame.MOUSEBUTTONUP:
				self.releasePosition = list(pygame.mouse.get_pos())
				self.releasePosition[0] = self.releasePosition[0] / self.widthPixelsPerVirtualPixel
				self.releasePosition[1] = self.releasePosition[1] / self.heightPixelsPerVirtualPixel
				self.holdingState = False
				self.shapes.sliceShapes([self.clickPosition, self.releasePosition])
			if event.type == pygame.MOUSEMOTION:
				self.previousPosition = self.currentPosition
				self.currentPosition = list(pygame.mouse.get_pos())
				self.currentPosition[0] = self.currentPosition[0] / self.widthPixelsPerVirtualPixel
				self.currentPosition[1] = self.currentPosition[1] / self.heightPixelsPerVirtualPixel

				self.a.position = self.currentPosition
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_SPACE:
					self.shapes.createShape(self.virtualWidth / 2, self.virtualHeight / 2)

				if event.key == pygame.K_z:
					b = self.space.point_query_nearest(self.currentPosition, 20, pymunk.ShapeFilter())
					if b != None:
						if self.c != None:
							self.space.remove(self.c)
						self.c = pymunk.DampedSpring(self.a, b[0].body, (0, 0), b[0].body.center_of_gravity, 10, 30, 0)
						self.space.add(self.c)

				if event.key == pygame.K_x:
						if self.c != None:
							self.space.remove(self.c)
							self.c = None


class Shape():

	def __init__(self, points = [], mVec = [0.0, 0.0], mVel = 1, ang = 0, aVel = 0):
		self.position = [0.0, 0.0]
		self.points = []
		self.setPoints(points)

		self.movementVector = mVec
		self.movementVelocity = mVel
		self.angle = ang
		self.angleVelocity = aVel

	def setPoints(self, points):
		self.points.clear()
		self.points.extend(points)
		self.setCenter()

	def setCenter(self):
		self.position = centerFromPoints(self.points)

	def movePointsTo(self, x, y):
		xDiff = x - self.position[0]
		yDiff = y - self.position[1]

		for i in range(len(self.points)):
			self.points[i][0] += xDiff
			self.points[i][1] += yDiff

		self.position[0] = x
		self.position[1] = y

	def rotatePoints(self, origin, angle):
		for i in range(len(self.points)):
			ox, oy = origin
			px, py = self.points[i]

			qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
			qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

			self.points[i] = [qx, qy]

		if origin != self.position:
			self.setCenter()

	def update(self, deltaTime):
		self.movementVector[1] += (1 * deltaTime)

		xDiff = (self.movementVector[0] * self.movementVelocity) * deltaTime
		yDiff = (self.movementVector[1] * self.movementVelocity) * deltaTime
		aDiff = self.angleVelocity * deltaTime

		for i in range(len(self.points)):
			self.points[i][0] += xDiff
			self.points[i][1] += yDiff

		self.position[0] += xDiff
		self.position[1] += yDiff

		self.angle += aDiff
		self.rotatePoints(self.position, aDiff)

	def findIntersections(self, line):
		intersections = []

		for i in range(-1, len(self.points) - 1, 1):
			pointLine = [self.points[i], self.points[i + 1]]
			intersection = lineSegmentIntersection(pointLine, line)

			if intersection != []:
				intersections.append([intersection, i])

		return intersections

	def sliceIntersections(self, intersections):
		newShapes = []
		points = self.points[:]
		offset = 0

		for i in range(len(intersections)):
			points.insert(intersections[i][1] + 1 + offset, intersections[i][0])
			intersections[i][1] = intersections[i][1] + (1 + offset)
			offset += 1

		for i in range(len(intersections)):
			if intersections[i][1] < intersections[(i + 1) % len(intersections)][1]:
				newPoints = points[intersections[i][1]: intersections[(i + 1) % len(intersections)][1] + 1]
			else:
				newPoints = points[intersections[i][1]:] + points[: intersections[(i + 1) % len(intersections)][1] + 1]

			centerPoint = centerFromPoints([intersections[i][0], intersections[(i + 1) % len(intersections)][0]])
			newPointsCenter = centerFromPoints(newPoints)
			angleBetweenCenters = angleBetweenPoints(centerPoint, newPointsCenter)
			newMVec = angleToVector(angleBetweenCenters)
			#newMVec[0] *= 1000
			#newMVec[1] *= 1000

			newShape = Shape(points = newPoints, mVec = newMVec, mVel = self.movementVelocity, ang = self.angle, aVel = self.angleVelocity)
			newShapes.append(newShape)

		return newShapes


class Shape2():

	def __init__(self, points, mass, pos, angle, angularVel, velocity, friction):
		self.boxBody = pymunk.Body(mass, 1666, pymunk.Body.DYNAMIC)
		self.boxBody.position = pos
		#self.boxBody.center_of_gravity = centerFromPoints(self.translateTo00(points))
		self.boxBody.center_of_gravity = centerFromPoints(points)

		self.boxBody.angle = angle
		self.boxBody.angular_velocity = angularVel
		self.boxBody.velocity = velocity

		#self.boxPoly = pymunk.Poly(self.boxBody, self.translateTo00(points))
		self.boxPoly = pymunk.Poly(self.boxBody, points)

		self.boxPoly.friction = friction
		self.boxPoly.elasticity = 0.4


	def getTopLeft(self, points):
		minX = 100000000
		minY = 100000000
		for point in points:
			if point[0] < minX:
				minX = point[0]
			if point[1] < minY:
				minY = point[1]

		return [minX, minY]

	def translateTo00(self, points):
		topLeft = self.getTopLeft(points)

		for i in range(len(points)):
			points[i][0] - topLeft[0]
			points[i][1] - topLeft[1]

		return points


	def addToSpace(self, space):
		space.add(self.boxBody, self.boxPoly)

	def findIntersections(self, line):
		intersections = []
		points = self.getPoints()

		for i in range(-1, len(points) - 1, 1):
			pointLine = [points[i], points[i + 1]]
			intersection = lineSegmentIntersection(pointLine, line)

			if intersection != []:
				intersections.append([intersection, i])

		return intersections

	def sliceIntersections(self, intersections):
		newShapes = []
		points = self.getPoints()
		offset = 0

		for i in range(len(intersections)):
			points.insert(intersections[i][1] + 1 + offset, intersections[i][0])
			intersections[i][1] = intersections[i][1] + (1 + offset)
			offset += 1

		for i in range(len(intersections)):
			if intersections[i][1] < intersections[(i + 1) % len(intersections)][1]:
				newPoints = points[intersections[i][1]: intersections[(i + 1) % len(intersections)][1] + 1]
			else:
				newPoints = points[intersections[i][1]:] + points[: intersections[(i + 1) % len(intersections)][1] + 1]

			centerPoint = centerFromPoints([intersections[i][0], intersections[(i + 1) % len(intersections)][0]])
			newPointsCenter = centerFromPoints(newPoints)
			angleBetweenCenters = angleBetweenPoints(centerPoint, newPointsCenter)
			newMVec = angleToVector(angleBetweenCenters)
			newMVec[0] *= 500
			newMVec[1] *= 500

			newShape = Shape2(newPoints[:], self.boxBody.mass, newPointsCenter[:], 0, -self.boxBody.angular_velocity, newMVec[:], self.boxPoly.friction)
			newShapes.append(newShape)

		return newShapes

	def getPoints(self):
		ps = [pos.rotated(self.boxBody.angle) + self.boxBody.position for pos in self.boxPoly.get_vertices()]

		return ps

	def getPosition(self):
		ps = [pos.rotated(self.boxBody.angle) + self.boxBody.position for pos in self.boxPoly.get_vertices()]
		return centerFromPoints(ps)


class Shapes():

	def __init__(self, space, width, height):
		self.shapes = []

		self.width = width
		self.height = height
		self.space = space

		self.didSlice = False

	def addShape(self, shape):
		if isinstance(shape, list):
			for s in shape:
				self.space.add(s.boxBody, s.boxPoly)
			self.shapes.extend(shape)
		else:
			self.shapes.append(shape)
			self.space.add(shape.boxBody, shape.boxPoly)

	def createShape(self, x, y):
		#s = Shape(self.generateCube(random.randint(self.width / 32, self.width / 8)), mVec = [random.uniform(-0.25, 0.25), -1], mVel = random.uniform(self.height / 2, self.height), aVel = random.uniform(-6, 6))
		s = Shape2(self.generateCube(random.randint(self.width / 32, self.width / 8)), 0.55, [x, y], 0, random.uniform(-10, 10), [random.uniform(-500, 500), -750], 0.25)
		self.addShape(s)

	def removeShape(self, shape):
		self.shapes.remove(shape)
		self.space.remove(shape.boxBody, shape.boxPoly)

	def sliceShapes(self, line):
		for shape in self.shapes[:]:
			intersections = shape.findIntersections(line)

			if len(intersections) > 1:
				newShapes = shape.sliceIntersections(intersections)

				self.removeShape(shape)
				self.addShape(newShapes)

				self.didSlice = True

	def update(self, deltaTime):
		for shape in self.shapes[:]:
			#shape.update(deltaTime)

			if shape.getPosition()[1] > self.height * 2:
				self.removeShape(shape)

		if self.didSlice:
			self.didSlice = False

	def getLines(self):
		lineArr = []
		for shape in self.shapes:
			lineArr.append(shape.getPoints())

		return lineArr

	def generateCube(self, scale):
		return [[0, 0], [1 * scale, 0], [1 * scale, 1 * scale], [0, 1 * scale]]

	def generateCube2(self, scale):
		return [[0, 0], [1 * scale, 0], [1 * scale, 1 * scale], [2 * scale, 1 * scale], [2 * scale, 0], [3 * scale, 0], [3 * scale, 2 * scale], [0, 2 * scale]]


if __name__ == '__main__':
	game().start()
