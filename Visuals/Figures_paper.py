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
import shapely.affinity
from shapely.geometry import Polygon
import shapely.plotting

tetros_dict = {
'L' : np.array([(0, 0), (0, 3), (1, 3), (1, 1), (2, 1), (2, 0)]),
'O' : np.array([(0, 0), (0, 2), (2, 2), (2, 0)]),
'T' : np.array([(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 1), (3, 1), (3, 0)]),
'I' : np.array([(0, 0), (0, 1), (4, 1), (4, 0)]),
'S' : np.array([(0, 0), (0, 1), (1, 1), (1, 2), (3, 2), (3, 1), (2, 1), (2, 0)]),
'Z' : np.array([(0, 2), (2, 2), (2, 1), (3, 1), (3, 0), (1, 0), (1, 1), (0, 1)]),
'J' : np.array([(0, 0), (0, 1), (1, 1), (1, 3), (2, 3), (2, 0)]),
}

class Tetromino:
    def __init__(self, constructor_vertices:list[tuple], scaler:float = 1, color = 'b') -> None:
        self.__constructor_vertices = constructor_vertices
        self.id = id
        self.scaler = scaler
        self.polygon = Polygon(constructor_vertices*scaler)
        self.color = color
        self.__angle = 0.0

    @property
    def center(self) -> tuple:
        center = self.polygon.centroid.coords.xy
        return np.array([center[0][0], center[1][0]])
    
    @property
    def vertices(self) -> np.array:
        vertices = self.polygon.exterior.coords.xy
        return np.array([vertices[0], vertices[1]]).T
    @property
    def constructor_vertices(self) -> np.array:
        return self.__constructor_vertices.tolist()
    @center.setter
    def center(self, new_center: tuple) -> None:
        self.polygon = shapely.affinity.translate(self.polygon, xoff=new_center[0] - self.center[0], yoff=new_center[1] - self.center[1], zoff=0.0)

    def rotate(self, angle: float) -> None:
        self.polygon = shapely.affinity.rotate(self.polygon, angle, origin='centroid', use_radians=False)
        self.__angle = (self.__angle + angle)%360

    def translate(self, direction) -> None:
        self.polygon = shapely.affinity.translate(self.polygon, xoff=direction[0], yoff=direction[1], zoff=0.0)
        

    def plot(self, plot=None, text=None, dtx=0, dty=0, hide_angle=False) -> None:
        # contour
        points = self.polygon.exterior.coords
        x_values, y_values = zip(*points)

        shapely.plotting.plot_polygon(self.polygon,
                                      add_points=False, 
                                      facecolor=self.color, 
                                      #edgecolor='black', 
                                      alpha=0.9, 
                                      zorder=3, 
                                      linewidth=0
                                      
                                      )

    
        # center
        center = self.polygon.centroid
        plot.plot(center.x, center.y, 'o', color='black', zorder=3)
        radius = 0.5

        if not hide_angle:
            # plot angle
            xangle = np.cos(np.deg2rad(self.angle))*radius*2
            yangle = np.sin(np.deg2rad(self.angle))*radius*2
            # print(xangle, yangle)
            plot.plot([center.x, center.x + xangle], [center.y,
                       center.y+yangle], color='black', zorder=3)
            plot.plot([center.x, center.x+radius*2],
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
                plot.text(center.x + xtext + dtx, center.y + ytext + dty,
                           text, color='black', zorder=4, fontsize=COLORS.FONT_SIZE)


            plot.gca().add_patch(arc)


    def set_angle(self, new_angle):
        angle = new_angle - self.__angle
        self.polygon = shapely.affinity.rotate(self.polygon, angle, origin='centroid', use_radians=False)
        self.__angle = new_angle
    
    @property
    def angle(self) -> float:
        return self.__angle
    
    def print_info(self) -> None:
        print('center: {}'.format(self.center))
        print('angle: {}'.format(self.__angle))

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


def contact_tiles(obj: Polygon, N: int, color):
    for i in range(N):
        for j in range(N):
            # if the object touchs the center of the tile
            if obj.contains(Point(i+0.5, j+0.5)):
                tile = Polygon([(i, j), (i+1, j), (i+1, j+1), (i, j+1)])
                shapely.plotting.plot_polygon(tile, 
                 add_points=False, 
                 facecolor='none',  # Set facecolor to 'none'
                 edgecolor=color,  # Specify the desired edgecolor
                 alpha=1,
                 zorder=2, 
                 linewidth=2)

    pass


class COLORS:
    import colors
    palette = colors.create_palette(4,normalize=True)
    colors.show_palette(palette, save=True, name='Images/Paper/Palette.png')

    OBJECT =        palette[1]
    TARGET =        palette[2]
    CONTACT_TILE =  palette[0]
    TARGET_TILE =   palette[3]
    GRID =          '#d3d3d3'
    FONT_SIZE = 12

def figure_environment():
    N = 10
    M = 8
    plt.figure(figsize=(3.5, 3.5))
    for i in range(N):
        for j in range(M):
            plt.plot([i, i+1], [j, j], COLORS.GRID)
            plt.plot([i, i], [j, j+1], COLORS.GRID)
            plt.plot([i, i+1], [j+1, j+1], COLORS.GRID)
            plt.plot([i+1, i+1], [j, j+1], COLORS.GRID)
            plt.plot(i+0.5, j+0.5, 'x', color=COLORS.GRID)

    resolution = 1.5

    symbol = 'Z'
    vertex = tetros_dict[symbol]
    tetrom = Tetromino(vertex, resolution, COLORS.OBJECT)
    tetrom.rotate(45)
    tetrom.translate([0.7, 0.4])
    tetrom.plot(plot=plt, text='$\\alpha$', dtx=0.3, dty=0.15)
    
    
    target = Tetromino(vertex, resolution, COLORS.TARGET)
    target.rotate(75)
    target.translate([N*0.55, M*0.49])
    target.plot(plot=plt, text='$\\beta$', dtx=0.25, dty=0.25)
    
    contact_tiles(tetrom.polygon, N, COLORS.CONTACT_TILE)
    contact_tiles(target.polygon, N, COLORS.TARGET_TILE)

    plt.plot([], [], 's', label="Object", color=COLORS.OBJECT)
    plt.plot([], [], 's', label="Target position", color=COLORS.TARGET)
    plt.plot([], [], 's', label="Contact tiles", color=COLORS.CONTACT_TILE)
    plt.plot([], [], 's', label="Target tiles", color=COLORS.TARGET_TILE)
    plt.plot([], [], 'x', label="Sensor", color=COLORS.GRID)
    plt.plot([], [], 'o', label="Center", color='black')


    plt.legend(framealpha=1, loc='upper left')
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

        tetrom = Tetromino(symbol, resolution, COLORS.OBJECT)
        target = Tetromino(symbol, resolution, COLORS.TARGET)

        tetrom.rotate(object_angle[0])
        tetrom.move_to(object_center[0][0]/TILE_SIZE,
                       object_center[0][1]/TILE_SIZE)
        tetrom.plot(plot=image)

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
        target.plot(plot=image)

        # Plot Final position
        tetrom.rotate(object_angle[-1])
        tetrom.move_to(X[-1], Y[-1])
        tetrom.plot(plot=image)

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
    tetrom = Tetromino(symbol, resolution, COLORS.OBJECT)
    tetrom.rotate(35)
    tetrom.move_to(t_center[0], t_center[1])
    tetrom.plot(text='$\\alpha$')
    contact_tiles(tetrom.polygon, N, COLORS.CONTACT_TILE)

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
    t = Tetromino('T', 2, COLORS.OBJECT)
    l = Tetromino('J', 1, COLORS.OBJECT)
    s = Tetromino('Z', 0.75, COLORS.OBJECT)
    i = Tetromino('I', 0.25, COLORS.OBJECT)
    o = Tetromino('O', 0.5, COLORS.OBJECT)

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
    contact_tiles(t.polygon, N, COLORS.CONTACT_TILE)
    contact_tiles(l.polygon, N, COLORS.CONTACT_TILE)
    contact_tiles(s.polygon, N, COLORS.CONTACT_TILE)
    contact_tiles(i.polygon, N, COLORS.CONTACT_TILE)
    contact_tiles(o.polygon, N, COLORS.CONTACT_TILE)

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



import sys

def figure_experiment_resolution():
    path = 'Data/Resolution'
    all_files = glob.glob(os.path.join(path, "*.bin"))
    datas = []

    for file in all_files:
        this_data = process_experiment_data(file)
        datas.append(this_data)

    angle_res, angle_means, angle_std_devs = get_errorbar_data(datas, x_axis='resolution', error_type='error_angle')
    position_res, position_means, position_std_devs = get_errorbar_data(datas, x_axis='resolution', error_type='error_position')

    side = 6
    fig, ax1 = plt.subplots(figsize=(side, side/3))
    import colors
    palette = colors.create_palette(2,normalize=True)

    #colors = [COLORS.CONTACT_TILE, COLORS.TARGET_TILE]
    ax1.plot(angle_res, angle_means, marker='o', c=palette[0])
    fillup = [m - s for m, s in zip(angle_means, angle_std_devs)]
    filldown = [m + s for m, s in zip(angle_means, angle_std_devs)]
    plt.fill_between(angle_res, fillup, filldown, alpha=0.2, color=palette[0])
    # Create a second y-axis sharing the same x-axis
    
    ax2 = ax1.twinx()
    ax2.plot(position_res, position_means, marker='s', c=palette[1])
    fillup = [m - s for m, s in zip(position_means, position_std_devs)]
    filldown = [m + s for m, s in zip(position_means, position_std_devs)]
    plt.fill_between(position_res, fillup, filldown, alpha=0.2, color=palette[1])


    ax2.set_ylabel('Position error $[tiles]$')
    ax1.set_ylabel('Angle error $[°]$')
    ax1.set_xlabel('Resolution')


    plt.plot([], [], 'o', label="Angle error", color=palette[0])
    plt.plot([], [], 's', label="Position error", color=palette[1])
 
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'Images/Experiments/resolution.png', dpi=300, bbox_inches='tight')
    #plt.show()
    
    #plot_errorbar_data(datas=datas, x_axis='resolution', error_type='error_angle', ylabel='Angle Error $[°]$')
    #plot_errorbar_data(datas=datas, x_axis='resolution',error_type='error_position', ylabel='Position Error $[tiles]$')


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
    resolutions = [3, 4, 5]
    print(resolutions)
    new_datas = [[data for data in nd if data['resolution'] == resolution] for resolution in resolutions]


    all_data = {}
    for res, data in zip(resolutions,new_datas):
        data_resolution = {}

        for run in data:
            dead_tiles = run['dead_tiles']
            error_angle = run['error_angle']
            error_position = run['error_position']

            if dead_tiles not in data_resolution:
                data_resolution[dead_tiles] = {'error_angle': [], 'error_position': []}
            else:
                data_resolution[dead_tiles]['error_angle'].append(error_angle)
                data_resolution[dead_tiles]['error_position'].append(error_position)



        dead_tiles = list(data_resolution.keys())
        angle_e = [np.mean(data_resolution[dead_tiles]['error_angle']) for dead_tiles in data_resolution.keys()]
        position_e = [np.mean(data_resolution[dead_tiles]['error_position']) for dead_tiles in data_resolution.keys()]
        angle_std = [np.std(data_resolution[dead_tiles]['error_angle']) for dead_tiles in data_resolution.keys()]
        position_std = [np.std(data_resolution[dead_tiles]['error_position']) for dead_tiles in data_resolution.keys()]

        all_data[res] = {'dead_tiles': dead_tiles, 
                         'error_angle': angle_e, 
                         'error_position': position_e, 
                         'angle_std': angle_std, 
                         'position_std': position_std}
    



    gridspec_kw = dict(
    height_ratios=(1, 1),
    hspace=0,
    )

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw=gridspec_kw)
    colors = plt.cm.plasma(np.linspace(0, 1, len(resolutions)))
    import colors
    palette = colors.create_palette(len(resolutions)+1,normalize=True)
    #remove second color
    palette.pop(1)
    #colors = [COLORS.CONTACT_TILE, COLORS.OBJECT, COLORS.GRID]
    #set size of figure
    fig.set_size_inches(6, 3)

    for i, res in enumerate(resolutions):
        error_angle = all_data[res]['error_angle']
        error_position = all_data[res]['error_position']
        dead_tiles = all_data[res]['dead_tiles']
        angle_std = all_data[res]['angle_std']
        position_std = all_data[res]['position_std']

        axes[0].plot(dead_tiles, error_angle, marker='o', label=f'Resolution = {res}', c=palette[i])
        fillup = [m - s for m, s in zip(error_angle, angle_std)]
        filldown = [m + s for m, s in zip(error_angle, angle_std)]
        axes[0].fill_between(dead_tiles, fillup, filldown, alpha=0.2, color=palette[i])

        axes[1].plot(dead_tiles, error_position, marker='o', label=f'Resolution = {res}', c=palette[i])
        fillup = [m - s for m, s in zip(error_position, position_std)]
        filldown = [m + s for m, s in zip(error_position, position_std)]
        axes[1].fill_between(dead_tiles, fillup, filldown, alpha=0.2, color=palette[i])


    axes[0].set_ylabel('Angle error\n[$°$]')
    axes[1].set_ylabel('Position error\n[$tiles$]')
    axes[1].set_xlabel('Faulty Tiles [%]')
    #axes[0].legend(loc='upper left')
    axes[1].legend(loc='upper left')
    axes[0].set_xticks(dead_tiles)
    axes[1].set_xticks(dead_tiles)
    axes[0].set_xticklabels([int(t * 100) for t in dead_tiles])
    axes[1].set_xticklabels([int(t * 100) for t in dead_tiles])

    plt.tight_layout()
    plt.savefig(f'Images/Experiments/faulty.png', dpi=300, bbox_inches='tight')

    #plt.show()
    sys.exit()
            
    



if __name__ == '__main__':
    figure_environment()
    #figure_trajectory()
    #figure_translation_vector()
    #figure_rotation_vector()
    #figure_resolution()
    
    figure_experiment_resolution()
    figure_experiment_faulty()
