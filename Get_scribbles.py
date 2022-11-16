"""
This script allows to annotate some pixels of the image by drawing lines.
Don't forget to indicate the number of regions and the name of the image
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

###########################
NB_CLASS = 2

im = np.array(Image.open('Images/lapin.jpeg'), dtype=np.uint8)
H, W = im.shape[0], im.shape[1]

Scribbles = {}
############################


def get_line(start, end):
    """
	This code comes from the following Git : 
	https://gist.github.com/Siyeong-Lee/fdacece959c6973603dbda955e45637b
	Bresenham's Line Algorithm
    Produces a list of tuples from start and end
    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    >>> print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
    """
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points

def get_entire_scribbles(Scribbles):
	Scribbles_line = {}
	Nb_class = len(Scribbles)
	for i in range(1,Nb_class +1):
		Scribbles_line['Class_'+str(i)] = []
		x,y = Scribbles['Class_'+str(i)][0], Scribbles['Class_'+str(i)][1]
		for ind in range(0,len(x),2):
			line = get_line((int(x[ind]),int(y[ind])),(int(x[ind+1]),int(y[ind+1])))
			Scribbles_line['Class_'+str(i)].extend(line[:-1])
	return Scribbles_line

def onclick(event):
	if event.button == 1:
		if event.dblclick:
			v = []
			for i in range(len(xs)):
				v.append((xs[i], ys[i]))
			v.append((xs[0], ys[0]))
		else:
			xs.append(event.xdata)
			ys.append(event.ydata)
			po.set_data(xs, ys)
			po.figure.canvas.draw_idle()
		if len(xs)==3: 
			scribs[0].extend(xs[:-1])
			scribs[1].extend(ys[:-1])
			del xs[:-1]
			del ys[:-1]
			

for i in range(1,NB_CLASS + 1):
	scribs = [[],[]]
	fig = plt.figure()
	plt.imshow(im)
	po, = plt.plot([], [])
	xs, ys = list(po.get_xdata()), list(po.get_ydata())
	connection_id = fig.canvas.mpl_connect('button_press_event', onclick)
	plt.title('scribble for class '+str(i))
	plt.show()
	scribs[0].extend(xs)
	scribs[1].extend(ys)
	Scribbles['Class_'+str(i)] = scribs

	#Scribbles['Class_'+str(i)] = [xs,ys]


line_scribbles = get_entire_scribbles(Scribbles)

np.save('Scrib/scrib.npy',line_scribbles)