import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.spatial import distance
from scipy.spatial import cKDTree
import math

import plotly.graph_objects as go

from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import euclidean

def get_neighbors(points, target_point):
    """
    Identify the neighboring points of a given target point within a Voronoi diagram. The neighbors are identified 
    based on the Voronoi ridges between the target point and its adjacent points.

    Args:
    points (np.ndarray): A numpy array of points [[x1, y1], [x2, y2], ...], including the target point.
    target_point (np.ndarray): A numpy array representing the target point [x, y].

    Returns:
    np.ndarray: A numpy array of the neighboring points to the target point including the target point.

    """
    target_idx = np.where((points == target_point).all(axis=1))[0][0]
    vor = Voronoi(points)
    neighbor_indices = [i for i in range(len(points)) if 
                vor.ridge_points[np.isin(vor.ridge_points, target_idx).any(1)].flatten().tolist().count(i) > 0]  
    vor_np_n = points[neighbor_indices]
    return vor_np_n

def order_polygon_points(points):
    """
    Order a numpy array representing a polygon's coordinates.

    Args:
    points (np.ndarray): A numpy array of points [[x1, y1], [x2, y2], ...] as the polygon's vertices.

    Returns:
    np.ndarray: The ordered coordinates of the polygon.
    """
    # Calculate centroid
    centroid_x = np.mean(points[:, 0])
    centroid_y = np.mean(points[:, 1])

    # Sort the points based on angle from centroid
    sorted_points = sorted(points, key=lambda point: math.atan2(point[1] - centroid_y, point[0] - centroid_x))

    return np.array(sorted_points)

def get_vertices(points, target_point):
    """
    Computes and orders the vertices of the Voronoi cell corresponding to a specified point.

    Args:
    points (np.ndarray): An ORDERED numpy array of points [[x1, y1], [x2, y2], ...] as the polygon's vertices.
    target_point (np.ndarray): A numpy array representing the target point [x, y].

    Returns:
    np.ndarray: An ordered numpy array of the vertices.
    """
    vor = Voronoi(points)
    vertices = vor.vertices
    
    target_point_idx = np.where((points == target_point).all(axis=1))[0][0]
    region_index = vor.point_region[target_point_idx]
    region_vertices = vor.regions[region_index]
    vertices = np.array([v for i,v in enumerate(vertices) if i in region_vertices])
    vertices = order_polygon_points(vertices)
    
    return vertices    

def get_intersections(points, x_value):
    """
    Find the intersection points of a polygon with a vertical line x = x_value.

    Args:
    points (np.ndarray): An ORDERED numpy array of points [[x1, y1], [x2, y2], ...] as the polygon's vertices, known
    that x_value is between the min and max x values of points.
    x_value (float): The x-value of the vertical line.

    Returns:
    numpy array: The intersection points with the line x = x_value.
    """
    intersections = []

    for i in range(len(points)):
        start, end = points[i], points[(i + 1) % len(points)]

        if (start[0] <= x_value and end[0] >= x_value) or (end[0] <= x_value and start[0] >= x_value):
            if start[0] == end[0]: 
                if start[0] == x_value:
                    intersections.append([start, end])
            else:
                t = (x_value - start[0]) / (end[0] - start[0])
                y_intersect = start[1] + t * (end[1] - start[1])
                intersections.append([x_value, y_intersect])

    if len(intersections) != 2:
        raise Exception("number of intersections not equal to two")

    return np.array(intersections)

def calculate_area(points):
    """
    Calculate the area of a polygon given its vertices using shoelace formula.
    
    Args:
    points (np.ndarray): An ORDERED numpy array of points [[x1, y1], [x2, y2], ...] as the polygon's vertices.

    Returns:
    float: The area of the polygon.
    """
    n = len(points)  
    area = 0.0

    for i in range(n):
        j = (i + 1) % n  
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]

    area = abs(area) / 2.0
    return area

def get_areas(points, target_point):
    """
    Calculates the areas of the sub polygons formed by dividing a polygon with a vertical line through a given point.

    Args:
    points (np.ndarray): A numpy array of points [[x1, y1], [x2, y2], ...] as the polygon's vertices, known that 
    the target_point is within the polygon.
    target_point (np.ndarray): A numpy array representing the target point [x, y].

    Returns:
    tuple: A tuple containing the area of the left polygon, the area of the right polygon, 
           vertices of the left polygon, and vertices of the right polygon.
    """
    intersections =  get_intersections(points, target_point[0])
    
    points_right = np.array([p for p in points if p[0] > target_point[0]])
    points_right = np.vstack([points_right, intersections])
    points_right = order_polygon_points(points_right)
    area_right = calculate_area(points_right)
    
    points_left = np.array([p for p in points if p[0] < target_point[0]])
    points_left = np.vstack([points_left, intersections])
    points_left = order_polygon_points(points_left)
    area_left = calculate_area(points_left)    
    
    return area_left, area_right, points_left, points_right

def plot_polygon(vertices, color='grey', alpha = 0.5):
    '''
    Fill a polygon in a plot.
    '''
    if len(vertices) > 0:
        x = vertices[:, 0].tolist() + [vertices[0, 0]]
        y = vertices[:, 1].tolist() + [vertices[0, 1]]
        return go.Scatter(x=x, y=y, fill='toself', fillcolor=color, opacity=alpha, 
                          mode='none', showlegend=False)

def plot_vor(points_nbs, hull_points, carrier_caught_xy, boundary_points, vertices, 
             vertices_left, vertices_right, offense_caught_xy_n):
    """
    Creates a plot of a Voronoi diagram with additional features including hulls, carriers, boundaries, and vertices.

    Args:
    points_nbs (np.ndarray): An array of points used to generate the Voronoi diagram.
    hull_points (np.ndarray): Points used to create a convex hull, if possible.
    carrier_caught_xy (np.ndarray): The position of the ball carrier.
    boundary_points (np.ndarray): Points that define the boundary of the area of interest.
    vertices (np.ndarray): Vertices of the Voronoi cell associated with the ball carrier.
    vertices_left (np.ndarray): Vertices of the left section of the divided Voronoi cell.
    vertices_right (np.ndarray): Vertices of the right section of the divided Voronoi cell.
    offense_caught_xy_n (np.ndarray): Positions of the offenders.

    Returns:
    None: The function creates and displays a plot.
    """
    fig = go.Figure()

    vor = Voronoi(points_nbs)

    bounding_box = np.array(
        [[-100,-100], [-100,200], [400,-100], [400,200]])

    # Create a Voronoi diagram
    vor = Voronoi(np.vstack([points_nbs, bounding_box]))

    # Plot Voronoi edges
    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            fig.add_trace(go.Scatter(x=vor.vertices[simplex, 0], y=vor.vertices[simplex, 1],
                                     mode='lines', line_color='grey', line_width=0.5, showlegend=False))

    # Separate defender points from carrier and boundary points
    defense_nbs_xy = np.array([p for i, p in enumerate(points_nbs) if 
                               not np.any((p == np.vstack([boundary_points, carrier_caught_xy])).all(axis=1))])


    # Plot vertices
    fig.add_trace(go.Scatter(x=vertices[:, 0], y=vertices[:, 1], mode='markers',
                             marker=dict(color='black', size=5), name='Vertices', showlegend=False))

    # Plot Convex Hull, if possible
    if len(np.unique(hull_points, axis=0)) >= 3:
        hull = ConvexHull(hull_points)
        for simplex in hull.simplices:
            fig.add_trace(go.Scatter(x=hull_points[simplex, 0], y=hull_points[simplex, 1], 
                                     mode='lines', line=dict(color='grey', dash='dot'), name='Convex Hull', showlegend=False))

    # Plot polygons for divided Voronoi cell
    fig.add_trace(plot_polygon(vertices_right, alpha=0.5))
    fig.add_trace(plot_polygon(vertices_left, alpha=0.2))

    # Add scatter plots for different points
    fig.add_trace(go.Scatter(x=defense_nbs_xy[:, 0], y=defense_nbs_xy[:, 1], mode='markers',
                             marker=dict(color='darkblue', size=10), name='Defenders'))
    fig.add_trace(go.Scatter(x=[carrier_caught_xy[0]], y=[carrier_caught_xy[1]], mode='markers',
                             marker=dict(color='sienna', size=10), name='Ball Carrier'))
    fig.add_trace(go.Scatter(x=offense_caught_xy_n[:, 0], y=offense_caught_xy_n[:, 1], mode='markers',
                             marker=dict(color='peru', size=7), name='Offenders'))

    points_plot = np.vstack([vertices, defense_nbs_xy])   
    max_y = np.max(points_plot[:, 1])
    min_y = np.min(points_plot[:, 1])
    max_x = np.max(points_plot[:, 0])
    min_x = np.min(points_plot[:, 0])
    ylim_top = max_y + 2 if max_y < 51.3 else 53.3
    ylim_bot = min_y - 2 if min_y > 2 else 0

    # Set layout
    fig.update_layout(plot_bgcolor='white', xaxis=dict(range=[min_x-1, max_x+2], showticklabels = False),
                      yaxis=dict(range=[ylim_bot-1, ylim_top-1], showticklabels = False),
                      showlegend=True, legend=dict(x=1, y=1, font = dict(size=13)),
                      dragmode=False, width=500, height=400,
                      margin=dict(t=4, b=0, r=2), )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, scaleanchor="x", scaleratio=1)    
    
    return fig

def speed_to_vector(s, diret):
    """
    Converts speed and direction into a 2D velocity vector.

    Parameters:
    s (float): Speed.
    diret (float): Direction in degrees (0-360).

    Returns:
    numpy.array: A 2D vector representing the speed and direction.
    """
    dir_radians = math.radians(diret)
    theta = math.radians(90) - dir_radians

    x = round(s * math.cos(theta),2)
    y = round(s * math.sin(theta),2)
    return np.array([x, y])

def calculate_MPD(df1, df2):
    """
    Calculates the minimum distance between two moving objects, based on the initial positions and velocities
    of the objects using data in df1 and df2.

    Parameters:
    df1 (DataFrame): A DataFrame containing 'x', 'y' (position coordinates),
                     'dir' (direction), and 's' (speed) of the first object.
    df2 (DataFrame): A DataFrame containing 'x', 'y' (position coordinates),
                     'dir' (direction), and 's' (speed) of the second object.

    Returns:
    float: The Minimum Possible Distance (MPD) between the two objects.
    """
    x_1, y_1, dir_1, s_1 = df1[['x', 'y', 'dir', 's']].iloc[0]
    x_2, y_2, dir_2, s_2 = df2[['x', 'y', 'dir', 's']].iloc[0]
    V1 = speed_to_vector(s_1, dir_1)
    V2 = speed_to_vector(s_2, dir_2)
    V_12 = V1 - V2

    xy_1 = np.array([x_1, y_1])
    xy_2 = np.array([x_2, y_2])

    cross_product = np.linalg.det(np.array([V_12, xy_1 - xy_2]))
    V_12_abs = np.linalg.norm(V_12)
    MPD = np.abs(cross_product) / V_12_abs

    return MPD

def plot_MPD(MPD_dict):
    sorted_MPD = sorted(MPD_dict.items(), key=lambda x: x[1])
    labels, values = zip(*sorted_MPD)
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation='h', width=0.4, marker_color='darkblue'
    ))
    fig.update_layout(xaxis = dict(tickfont=dict(size = 12), dtick=1),
                      yaxis_tickfont=dict(size=14),
                      yaxis=dict(ticks="outside", tickcolor='white', ticklen=10),
                      yaxis_autorange="reversed", width=550, height=250, margin=dict(t=0, r =0),
                      dragmode=False)

    fig.update_yaxes(showgrid=False, scaleanchor="x", scaleratio=1)

    return fig