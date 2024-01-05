import glob
import os

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import svgwrite
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Arc, Circle, Wedge
from shapely import affinity
from shapely.geometry import Point, Polygon
import ast
import matplotlib.cm as cm
import math


class Tetromino():
    def __init__(self, symbol: str, resolution: float, color: str) -> None:
        self.color = color
        self.symbol = symbol
        self.resolution = resolution
        self.polygon = Polygon(self._create_polygob())
        self.angle = 0
        pass

    def rotate(self, angle: float) -> None:
        to_rot = angle - self.angle
        self.polygon = affinity.rotate(
            self.polygon, to_rot, origin=self.polygon.centroid)
        self.angle = angle
        pass

    def move_to(self, x: float, y: float) -> None:
        x = x - self.polygon.centroid.x
        y = y - self.polygon.centroid.y
        self.polygon = affinity.translate(self.polygon, xoff=x, yoff=y)
        pass

    def _create_polygob(self) -> Polygon:
        image_path = f"Images\Tetrominos\{self.symbol}.png"
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        edged = cv2.Canny(gray, 30, 200)
        contours, hierarchy = cv2.findContours(
            edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours, _ = cv2.findContours(
            threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [np.squeeze(contour) for contour in contours]
        max_ = max(max([max(contour[:, 0]) for contour in contours]), max(
            [max(contour[:, 1]) for contour in contours]))
        contours = [contour/max_ for contour in contours]

        scales = {
            'I': 4,
            'J': 3,
            'L': 3,
            'O': 2,
            'S': 3,
            'T': 3,
            'Z': 3,
        }

        long_side = scales[self.symbol]*self.resolution
        contours = [contour*long_side for contour in contours]
        polygon = Polygon(contours[0]).simplify(0.25, preserve_topology=True)
        return polygon

    def plot(self, where=None, text=None, dtx=0, dty=0, hide_angle=False) -> None:
        # contour
        points = self.polygon.exterior.coords
        x, y = zip(*points)

        if not where:
            where = plt
        where.fill(x, y, self.color, alpha=0.9, zorder=3)

        # center
        center = self.polygon.centroid
        where.plot(center.x, center.y, 'o', color='black', zorder=3)
        radius = self.resolution/2

        if not hide_angle:
            # plot angle

            xangle = np.cos(np.deg2rad(self.angle))*radius*2
            yangle = np.sin(np.deg2rad(self.angle))*radius*2
            # print(xangle, yangle)
            where.plot([center.x, center.x + xangle], [center.y,
                       center.y+yangle], color='black', zorder=3)
            where.plot([center.x, center.x+radius*2],
                       [center.y, center.y], color='black', zorder=3)

            # plot arch
            ang = self.angle
            if ang > 0:
                arc = Arc((center.x, center.y), radius*2, radius*2,
                          theta1=0, theta2=ang, color='black', zorder=3)
            else:
                arc = Arc((center.x, center.y), radius*2, radius*2,
                          theta1=ang, theta2=0, color='black', zorder=3)

            # plt text if needed
            if text:
                # plot text between center and angle
                xtext = np.cos(np.deg2rad(self.angle/2))*radius/2
                ytext = np.sin(np.deg2rad(self.angle/2))*radius/2
                where.text(center.x + xtext + dtx, center.y + ytext + dty,
                           text, color='black', zorder=3, fontsize=COLORS.FONT_SIZE)

            # plt.gca().add_patch(arc)
            # where.gca().add_patch(arc)
            if where == plt:
                where.gca().add_patch(arc)
            else:
                where.add_patch(arc)

    def plot_symbol_resolution(self, text, dx=0, dy=0):
        # plot a text over the center of the polygon
        center = self.polygon.centroid
        plt.text(center.x + dx, center.y + dy, text, color='black',
                 zorder=3, fontsize=COLORS.FONT_SIZE)

    def plot_contour(self, where=None, color: str = 'black') -> None:
        if not where:
            where = plt.gca()
        points = self.polygon.exterior.coords
        x, y = zip(*points)
        where.plot(x, y, color, zorder=3, linewidth=2, alpha=0.2)

    def get_svg_path(self):
        points = self.polygon.exterior.coords
        svg_document = svgwrite.Drawing(filename='polygon.svg')
        svg_document.add(svg_document.polygon(points=points, fill=self.color))
        svg_document.save()
        return svg_document.tostring()


def draw_grid(N,M, where=None):
    # Plot grid
    if not where:
        where = plt.gca()
    for i in range(N):
        for j in range(M):
            where.plot([i, i+1], [j, j], COLORS.GRID)
            where.plot([i, i], [j, j+1], COLORS.GRID)
            where.plot([i, i+1], [j+1, j+1], COLORS.GRID)
            where.plot([i+1, i+1], [j, j+1], COLORS.GRID)

            where.plot(i+0.5, j+0.5, 'x', color=COLORS.GRID)


def contact_tiles(obj: Polygon, N: int, color: str = 'black'):
    for i in range(N):
        for j in range(N):
            # if the object touchs the center of the tile
            if obj.contains(Point(i+0.5, j+0.5)):
                # draw the contour of the tile
                plt.plot([i, i+1], [j, j], color, zorder=2, linewidth=2)
                plt.plot([i, i], [j, j+1], color, zorder=2, linewidth=2)
                plt.plot([i, i+1], [j+1, j+1], color, zorder=2, linewidth=2)
                plt.plot([i+1, i+1], [j, j+1], color, zorder=2, linewidth=2)
    pass


class COLORS:
    TETROMINO = 'lightblue'
    TARGET = 'orange'
    CONTACT = 'blue'
    TARGET_TILE = 'red'
    GRID = 'lightgray'
    FONT_SIZE = 12


def figure_environment():

    N = 10
    M = 8
    
    plt.figure(figsize=(3.5, 3.5))
    draw_grid(N, M)
    resolution = 1.5

    symbol = 'S'
    tetrom = Tetromino(symbol, resolution, COLORS.TETROMINO)
    target = Tetromino(symbol, resolution, COLORS.TARGET)
    tetrom.rotate(-50)
    tetrom.move_to(N*0.3, M*0.25)
    tetrom.plot(text='$\\alpha$', dtx=0.5, dty=-0.3)
    target.rotate(160)

    target.move_to(N*0.75, M*0.7)
    target.plot(text='$\\beta$', dtx=-0.1, dty=-0.1)
    contact_tiles(tetrom.polygon, N, COLORS.CONTACT)
    contact_tiles(target.polygon, N, COLORS.TARGET_TILE)

    plt.plot([], [], 's', label="Object", color=COLORS.TETROMINO)
    plt.plot([], [], 's', label="Target", color=COLORS.TARGET)
    plt.plot([], [], 's', label="Contact tiles", color=COLORS.CONTACT)
    plt.plot([], [], 's', label="Target tiles", color=COLORS.TARGET_TILE)
    plt.plot([], [], 'x', label="Sensor", color=COLORS.GRID)
    plt.plot([], [], 'o', label="Center", color='black')


    plt.legend()
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig('Images/Paper/Environment.png', dpi=300, bbox_inches='tight')
    # plt.show()


def figure_trajectory():
    path = 'Data/Trajectories'
    all_files = glob.glob(os.path.join(path, "*.bin"))

    symbol_files = [file for file in all_files if 'symbol' in file]
    print(symbol_files)
    selected_symbols = ["I", "T", "O", "L", "J", "S", "Z"]
    selected_symbols = ["I", "T", "O", "Z"]
    symbol_files = [file for file in symbol_files if file.split(
        '_')[-1].split('.bin')[0] in selected_symbols]
    print(symbol_files)

    fig, ax = plt.subplots(1, len(selected_symbols))  # , figsize=(10, 10))

    for index, file in enumerate(symbol_files):
        image = ax[index]
        plt.gca().set_aspect('equal', adjustable='box')
        max_x = 0
        max_y = 0
        min_x = np.inf
        min_y = np.inf

        symbol = file.split('_')[-1].split('.bin')[0]

        with open(file, 'rb') as f:
            experiment_data = np.load(f, allow_pickle=True).item()

        N = experiment_data['N']
        TILE_SIZE = experiment_data['TILE_SIZE']
        resolution = experiment_data['resolution']
        symbol = experiment_data['symbol']
        data = experiment_data['data']

        object_center = data['object_center'].to_numpy()
        object_angle = data['object_angle'].to_numpy()
        target_center = data['target_center'].to_numpy()

        # remove the first element
        object_center = object_center[1:]
        object_angle = object_angle[1:]
        target_center = target_center[1:]

        # divide it bt tile_size
        target_center = (target_center[0][0] /
                         TILE_SIZE, target_center[0][1]/TILE_SIZE)
        target_angle = float(data['target_angle'].to_numpy()[0])

        tetrom = Tetromino(symbol, resolution, COLORS.TETROMINO)
        target = Tetromino(symbol, resolution, COLORS.TARGET)

        tetrom.rotate(object_angle[0])
        tetrom.move_to(object_center[0][0]/TILE_SIZE,
                       object_center[0][1]/TILE_SIZE)
        tetrom.plot(where=image)

        # Plot trajectory
        X = [point[0]/TILE_SIZE for point in object_center]
        Y = [point[1]/TILE_SIZE for point in object_center]

        for x, y, angle, i in zip(X, Y, object_angle, range(len(X))):
            color = cm.jet(i/len(X))
            tetrom.rotate(angle)
            tetrom.move_to(x, y)
            tetrom.plot_contour(where=image)

            points = tetrom.polygon.exterior.coords.xy
            x = points[0]
            y = points[1]

            max_x = max(max(x), max_x)
            max_y = max(max(y), max_y)
            min_x = min(min(x), min_x)
            min_y = min(min(y), min_y)

        # Plot target
        target.rotate(target_angle)
        target.move_to(target_center[0], target_center[1])
        target.plot(where=image)

        # Plot Final position
        tetrom.rotate(object_angle[-1])
        tetrom.move_to(X[-1], Y[-1])
        tetrom.plot(where=image)

        # cealing and floor of min and max on number divisible by TILE_SIZE
        max_x = np.ceil(max_x)
        max_y = np.ceil(max_y)
        min_x = np.floor(min_x)
        min_y = np.floor(min_y)

        draw_grid(N, where=image)

        title = symbol
        if symbol == 'S':
            title = 'Z'
        if symbol == 'Z':
            title = 'S'
        if symbol == 'J':
            title = 'L'
        if symbol == 'L':
            title = 'J'

        image.set_title(f'Shape {title}')

        image.set_xlim(min_x, max_x)
        image.set_ylim(min_y, max_y)

        image.set_aspect('equal', adjustable='box')
        image.set_axis_off()

    plt.axis('off')
    plt.tight_layout()
    plt.savefig('Images/Paper/Trajectory.png', dpi=300, bbox_inches='tight')
    # plt.show()


def draw_square(x, y, color, contour=False):
    plt.gca().add_patch(plt.Rectangle((x, y), 1, 1, color=color, fill=True, alpha=0.5))
    plt.plot(x+0.5, y+0.5, 'x', color=COLORS.GRID, zorder=3)
    if contour:
        plt.gca().add_patch(plt.Rectangle((x, y), 1, 1, color='black',
                                          fill=False, alpha=1, linewidth=2, zorder=2))


def draw_arrow(from_x, from_y, to_x, to_y, color, text=None, delta_x=0, delta_y=0):
    head_length = 0.1
    dx = to_x - from_x
    dy = to_y - from_y
    length = np.sqrt(dx**2 + dy**2)
    if length > 0:
        dx = dx - head_length*dx/length
        dy = dy - head_length*dy/length
        plt.arrow(from_x, from_y, dx, dy, head_width=0.05,
                  head_length=head_length, color=color, zorder=3)
        if text:
            mid_x = from_x + dx/2
            mid_y = from_y + dy/2
            plt.text(mid_x + delta_x, mid_y + delta_y, text, color='black',
                     ha='center', va='center', zorder=4, fontsize=COLORS.FONT_SIZE)


def figure_translation_vector():
    # set image size
    fig = plt.figure(figsize=(3, 3))
    x_0 = 0.5
    y_0 = -0.1

    draw_square(x_0, y_0, 'white', contour=True)
    draw_square(x_0, y_0+1, COLORS.TARGET_TILE)
    draw_square(x_0, y_0-1, COLORS.GRID)
    draw_square(x_0-1, y_0, COLORS.GRID)
    draw_square(x_0+1, y_0, COLORS.TARGET_TILE)

    plt.plot(0, 0, 'x', color='black')
    plt.text(0, -0.1, '$(0,0)$', color='black', ha='center',
             va='center', zorder=4, fontsize=COLORS.FONT_SIZE)

    draw_arrow(0, 0, x_0+0.5, y_0+1.5, 'darkgrey',
               text='$\\vec{p_{tn_1}}$', delta_x=- 0.2, delta_y=+ 0.1)
    draw_arrow(0, 0, x_0+1.5, y_0+0.5, 'darkgrey',
               text='$\\vec{p_{tn_2}}$', delta_y=- 0.15)
    draw_arrow(0, 0, x_0+0.5, y_0+0.5, 'darkgrey',
               text='$\\vec{p_t}$', delta_x=0.2, delta_y=0.1)

    avg_x = (x_0+0.5 + x_0+1.5)/2
    avg_y = (y_0+1.5 + y_0+0.5)/2
    draw_arrow(0, 0, avg_x, avg_y, 'darkgrey',
               text='$\overline{\\vec{p_{tn}}}$', delta_x=1, delta_y=0.7)

    draw_arrow(x_0+0.5, y_0+0.5, avg_x, avg_y, 'black',
               text='$\\vec{v_t} = \overline{\\vec{p_{tn}}} - \\vec{p_t}$', delta_x=0.6, delta_y=0)

    plt.axis('off')
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('Images/Paper/Translation.png', dpi=300, bbox_inches='tight')

    # plt.show()
    pass


def figure_rotation_vector():
    fig = plt.figure(figsize=(3, 3))
    N = 3
    draw_grid(N)
    resolution = 1
    t_center = (1.6, 1.6)
    symbol = 'T'
    tetrom = Tetromino(symbol, resolution, COLORS.TETROMINO)
    tetrom.rotate(35)
    tetrom.move_to(t_center[0], t_center[1])
    tetrom.plot(text='$\\alpha$')
    contact_tiles(tetrom.polygon, N, COLORS.CONTACT)

    example_tile = (2.5, 0.5)
    r = (t_center[0] - example_tile[0], t_center[1] - example_tile[1])
    draw_arrow(example_tile[0], example_tile[1], t_center[0],
               t_center[1], 'darkgrey', text='$\\vec{r}$', delta_x=- 0.2)
    perpendicular = (-r[1], r[0])
    draw_arrow(t_center[0], t_center[1], t_center[0] + perpendicular[0], t_center[1] +
               perpendicular[1], 'darkgrey', text='$\\vec{p}$', delta_x=0.2, delta_y=-0.05)
    unitary_perpendicular = (perpendicular[0]/np.linalg.norm(
        perpendicular), perpendicular[1]/np.linalg.norm(perpendicular))
    unitary_perpendicular = (
        unitary_perpendicular[0]*0.5, unitary_perpendicular[1]*0.5)

    text = f'$\\vec{{v_r}} = \\vec{{p}}.err.k$'

    draw_arrow(example_tile[0], example_tile[1], example_tile[0] + unitary_perpendicular[0],
               example_tile[1] + unitary_perpendicular[1], 'black', text=text, delta_x=-0.35, delta_y=0.3)

    # add square centered in example_tile
    draw_square(2, 0, 'white', contour=True)
    plt.axis('off')
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('Images/Paper/Rotation.png', dpi=300, bbox_inches='tight')
    pass


def figure_resolution():

    N = 6
    t = Tetromino('T', 2, COLORS.TETROMINO)
    l = Tetromino('J', 1, COLORS.TETROMINO)
    s = Tetromino('Z', 0.75, COLORS.TETROMINO)
    i = Tetromino('I', 0.25, COLORS.TETROMINO)
    o = Tetromino('O', 0.5, COLORS.TETROMINO)

    xl = l.polygon.centroid.x
    yl = l.polygon.centroid.y

    l.move_to(6 - xl, 6 - yl)
    l.rotate(180)

    s.move_to(4.8, 1.3)
    s.rotate(-45)

    i.move_to(2.8, 1)
    i.rotate(-30)

    o.move_to(2.7, 5.2)

    s.plot(hide_angle=True)
    t.plot(hide_angle=True)
    l.plot(hide_angle=True)
    i.plot(hide_angle=True)
    o.plot(hide_angle=True)

    draw_grid(N)
    contact_tiles(t.polygon, N, COLORS.CONTACT)
    contact_tiles(l.polygon, N, COLORS.CONTACT)
    contact_tiles(s.polygon, N, COLORS.CONTACT)
    contact_tiles(i.polygon, N, COLORS.CONTACT)
    contact_tiles(o.polygon, N, COLORS.CONTACT)

    s.plot_symbol_resolution('$Shape = S$' + '\n' + '$Res = 0.75$', -0.6, 0.2)
    t.plot_symbol_resolution('$Shape = T$' + '\n' + '$Res = 2$', -0.6, 0.2)
    l.plot_symbol_resolution('$Shape = L$' + '\n' + '$Res = 1$', -0.6, 0.2)
    i.plot_symbol_resolution('$Shape = I$' + '\n' + '$Res = 0.25$', -0.6, 0.2)
    o.plot_symbol_resolution('$Shape = O$' + '\n' + '$Res = 0.5$', -0.6, 0.2)

    plt.axis('off')
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('Images/Paper/Resolutions.png', dpi=300, bbox_inches='tight')

    pass




def figure_experiment_faulty():
    path = 'Data/Faulty'
    all_files = glob.glob(os.path.join(path, "*.bin"))
    nd = []
    for file in all_files:
        with open(file, 'rb') as f:
            experiment_data = np.load(f, allow_pickle=True).item()

        N = experiment_data['N']
        TILE_SIZE = experiment_data['TILE_SIZE']
        symbol = experiment_data['symbol']
        dead_tiles = experiment_data['dead_tiles']
        d = experiment_data['data']
        resolution = experiment_data['resolution']

        final_angle = d['object_angle'].iloc[-1]
        target_angle = d['target_angle'].iloc[-1]
        error_angle = abs(final_angle - target_angle)
        if error_angle > 180:
            error_angle = 360 - error_angle
        # print(final_angle, target_angle, error)

        final_position = np.array(d['object_center'].iloc[-1])
        target_position = np.array(d['target_center'].iloc[-1])
        error_position = np.linalg.norm(
            final_position - target_position)/TILE_SIZE

        this_data = {'N': N, 'TILE_SIZE': TILE_SIZE, 'dead_tiles': dead_tiles, 'resolution': resolution,
                     'symbol': symbol, 'data': d, 'error_angle': error_angle, 'error_position': error_position}
        nd.append(this_data)

    # get unique resolutions
    resolutions = list(set([data['resolution'] for data in nd]))
    new_datas = []
    for resolution in resolutions:
        new_datas.append(
            [data for data in nd if data['resolution'] == resolution])

    for typeerror in ['error_angle', 'error_position']:
        #change size of plt
        plt.figure(figsize=(6, 3))
        n_colors = len(resolutions)
        colors = plt.cm.plasma(np.linspace(0, 1, n_colors))
        for res, nd, color in zip(resolutions, new_datas, colors):
            
            grouped_data = {}
            for d in nd:
                perc = d['dead_tiles']
                error = d[typeerror]
                if perc not in grouped_data:
                    grouped_data[perc] = [error]
                else:
                    grouped_data[perc].append(error)
            
            percents = []
            means = []
            std_devs = []
            
            for perc, errors in grouped_data.items():
                percents.append(perc)
                means.append(np.mean(errors))
                std_devs.append(np.std(errors))
            
            
            plt.plot(percents, means, marker='o', label='$Resolution = {}$'.format(res), c = color)
            fillup = [m-s for m, s in zip(means, std_devs)]
            filldown = [m+s for m, s in zip(means, std_devs)]
            plt.fill_between(percents, fillup, filldown, alpha=0.3, color = color)


        plt.legend()
        #position of the legend
        plt.legend(loc='upper left')
        ticks = np.arange(0, 1, 0.1)
        labels = [int(t*100) for t in ticks]
        plt.xticks(ticks, labels=labels)
        
        plt.xlabel('Faulty Tiles [%]')        
        if typeerror == 'error_angle':
            plt.ylabel('Angle Error $[°]$')
        else:
            plt.ylabel('Position Error $[tiles]$')
        # y log scale
        #plt.yscale('log')
        #save figure
        plt.tight_layout()
        plt.savefig(f'Images/Experiments/faulty_{typeerror}.png', dpi=300, bbox_inches='tight')
        plt.clf()
        #plt.show()

    return



import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def load_experiment_data(file):
    with open(file, 'rb') as f:
        experiment_data = np.load(f, allow_pickle=True).item()
    return experiment_data

def calculate_angle_error(final_angle, target_angle):
    error_angle = abs(final_angle - target_angle)
    if error_angle > 180:
        error_angle = 360 - error_angle
    return error_angle

def calculate_position_error(final_position, target_position, TILE_SIZE):
    error_position = np.linalg.norm(final_position - target_position) / TILE_SIZE
    return error_position

def process_experiment_data(file):
    experiment_data = load_experiment_data(file)
    N = experiment_data['N']
    TILE_SIZE = experiment_data['TILE_SIZE']
    resolution = experiment_data['resolution']
    data = experiment_data['data']

    final_angle = data['object_angle'].iloc[-1]
    target_angle = data['target_angle'].iloc[-1]
    error_angle = calculate_angle_error(final_angle, target_angle)

    final_position = np.array(data['object_center'].iloc[-1])
    target_position = np.array(data['target_center'].iloc[-1])
    error_position = calculate_position_error(final_position, target_position, TILE_SIZE)

    return {'N': N, 'TILE_SIZE': TILE_SIZE, 'resolution': resolution, 'data': data,
            'error_angle': error_angle, 'error_position': error_position}

def plot_errorbar_data(datas, x_axis: str, error_type: str, ylabel: str):
    resolutions, means, std_devs = get_errorbar_data(datas, x_axis=x_axis, error_type=error_type)

    plt.figure(figsize=(6, 3))
    colors = plt.cm.plasma(np.linspace(0, 1, 3))

    plt.plot(resolutions, means, marker='o', c=colors[0])
    fillup = [m - s for m, s in zip(means, std_devs)]
    filldown = [m + s for m, s in zip(means, std_devs)]
    plt.fill_between(resolutions, fillup, filldown, alpha=0.2, color=colors[0])

    plt.xticks(resolutions)
    plt.xticks(rotation=45)

    plt.xlabel('Resolution')
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f'Images/Experiments/resolution_{error_type}.png', dpi=300, bbox_inches='tight')
    plt.clf()

def get_errorbar_data(datas, x_axis: str, error_type: str):
    grouped_data = {}
    for data in datas:
        x = data[x_axis]
        error = data[error_type]
        grouped_data.setdefault(x, []).append(error)

    xs = []
    means = []
    std_devs = []
    for x, errors in grouped_data.items():
        xs.append(x)
        means.append(np.mean(errors))
        std_devs.append(np.std(errors))

    return xs, means, std_devs

def figure_experiment_resolution():
    path = 'Data/Resolution'
    all_files = glob.glob(os.path.join(path, "*.bin"))
    datas = []

    for file in all_files:
        this_data = process_experiment_data(file)
        datas.append(this_data)

    plot_errorbar_data(datas=datas, x_axis='resolution',
                       error_type='error_angle', ylabel='Angle Error $[°]$')
    plot_errorbar_data(datas=datas, x_axis='resolution',
                       error_type='error_position', ylabel='Position Error $[tiles]$')


def process_faulty_experiment_data(file):
    experiment_data = load_experiment_data(file)
    N = experiment_data['N']
    TILE_SIZE = experiment_data['TILE_SIZE']
    symbol = experiment_data['symbol']
    dead_tiles = experiment_data['dead_tiles']
    d = experiment_data['data']
    resolution = experiment_data['resolution']

    final_angle = d['object_angle'].iloc[-1]
    target_angle = d['target_angle'].iloc[-1]
    error_angle = calculate_angle_error(final_angle, target_angle)

    final_position = np.array(d['object_center'].iloc[-1])
    target_position = np.array(d['target_center'].iloc[-1])
    error_position = calculate_position_error(final_position, target_position, TILE_SIZE)

    return {'N': N, 'TILE_SIZE': TILE_SIZE, 'dead_tiles': dead_tiles, 'resolution': resolution,
            'symbol': symbol, 'data': d, 'error_angle': error_angle, 'error_position': error_position}

def figure_experiment_faulty():
    path = 'Data/Faulty'
    all_files = glob.glob(os.path.join(path, "*.bin"))
    nd = [process_faulty_experiment_data(file) for file in all_files]

    resolutions = list(set([data['resolution'] for data in nd]))
    new_datas = [[data for data in nd if data['resolution'] == resolution] for resolution in resolutions]

    for typeerror in ['error_angle', 'error_position']:
        plt.figure(figsize=(6, 2))
        n_colors = len(resolutions)
        colors = plt.cm.plasma(np.linspace(0, 1, n_colors))
        
        for res, nd, color in zip(resolutions, new_datas, colors):
            grouped_data = {}
            for d in nd:
                perc = d['dead_tiles']
                error = d[typeerror]
                grouped_data.setdefault(perc, []).append(error)
            
            percents = []
            means = []
            std_devs = []
            
            for perc, errors in grouped_data.items():
                percents.append(perc)
                means.append(np.mean(errors))
                std_devs.append(np.std(errors))
            
            plt.plot(percents, means, marker='o', label=f'Resolution = {res}', c=color)
            fillup = [m - s for m, s in zip(means, std_devs)]
            filldown = [m + s for m, s in zip(means, std_devs)]
            plt.fill_between(percents, fillup, filldown, alpha=0.3, color=color)

        plt.legend(loc='upper left')
        ticks = np.arange(0, 1, 0.1)
        labels = [int(t * 100) for t in ticks]
        plt.xticks(ticks, labels=labels)
        
        plt.xlabel('Faulty Tiles [%]')
        plt.ylabel('Angle Error [$°$]' if typeerror == 'error_angle' else 'Position Error [$tiles$]')
        plt.tight_layout()
        plt.savefig(f'Images/Experiments/faulty_{typeerror}.png', dpi=300, bbox_inches='tight')
        plt.clf()




if __name__ == '__main__':
    #figure_environment()
    #figure_trajectory()
    #figure_translation_vector()
    #figure_rotation_vector()
    #figure_resolution()
    
    #figure_experiment_resolution()
    #figure_experiment_faulty()
