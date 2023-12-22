from shapely.geometry import Polygon, LineString, MultiPolygon
import cv2
import numpy as np
from scipy.optimize import minimize
from amftrack.pipeline.functions.image_processing.experiment_util import make_full_image
from amftrack.util.geometry import create_polygon


# Your existing create_polygon function here...
# ...


def find_intersection(p1, p2, point_on_line, direction):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = point_on_line
    dx, dy = direction

    denom = dx * (y2 - y1) - dy * (x2 - x1)

    if denom == 0:
        return None

    t = ((x1 - x3) * dy - (y1 - y3) * dx) / denom
    u = -((x1 - x3) * (y2 - y1) - (y1 - y3) * (x2 - x1)) / denom

    if 0 <= t <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return np.array([x, y])
    else:
        return None


def slice_polygon(vertices, points_on_lines, direction):
    intersections = []

    for point in points_on_lines:
        for i in range(len(vertices) - 1):
            p1 = vertices[i]
            p2 = vertices[i + 1]

            intersection = find_intersection(p1, p2, point, direction)

            if intersection is not None:
                intersections.append(intersection)

    return np.array(intersections)


def interpolate_edge_points(edge1, edge2, t_values):
    interpolated_points = np.array(
        [(1 - t) * np.array(edge1) + t * np.array(edge2) for t in t_values]
    )
    return interpolated_points


def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x4 - x3
    dy2 = y4 - y3
    det = dx1 * dy2 - dx2 * dy1
    if det == 0:
        return None  # Lines are parallel or coincident
    t1 = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / det
    t2 = ((x3 - x1) * dy1 - (y3 - y1) * dx1) / det
    if 0 <= t1 <= 1 and 0 <= t2 <= 1:
        # Intersection point is within both line segments
        intersection_x = x1 + t1 * dx1
        intersection_y = y1 + t1 * dy1
        return (intersection_x, intersection_y)
    return None  # No intersection


def get_polygons(center_x, center_y, angle, scale):
    # center_x, center_y = 0, 0
    # angle = 0  # Insert angle in degrees
    vertices, R, t = create_polygon(center_x, center_y, angle, scale)
    # Extract the endpoints of the straight edge
    edge1 = vertices[-1]
    edge2 = vertices[-2]

    # Interpolate points along the straight edge
    shift = 17.5 / 82 / 2
    t_values = [0.5 - 3 * shift, 0.5 - shift, 0.5 + shift, 0.5 + 3 * shift]
    points_on_lines = interpolate_edge_points(edge1, edge2, t_values)

    # Calculate the direction of the bisecting radius
    direction = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])

    # Calculate perpendicular direction
    perpendicular_direction = np.array([direction[1], -direction[0]])

    # Define distances above the straight edge
    distances = [10.50, 22, 33.50]  # Adjust these distances
    distances = [distance * scale for distance in distances]
    # Points for perpendicular lines
    points_on_perpendicular_lines = []
    point = points_on_lines[0]
    for distance in distances:
        new_point = point + distance * direction
        points_on_perpendicular_lines.append(new_point)

    line_points, directions = np.concatenate(
        (points_on_lines, points_on_perpendicular_lines)
    ), [direction] * len(points_on_lines) + [perpendicular_direction] * len(
        points_on_perpendicular_lines
    )
    extension_length = 80  # Adjust this based on your plot size
    extended_line_segments = [
        (
            point[0] - direction[0] * extension_length,
            point[1] - direction[1] * extension_length,
            point[0] + direction[0] * extension_length,
            point[1] + direction[1] * extension_length,
        )
        for point, direction in zip(line_points, directions)
    ]

    # Find intersections between lines
    line_intersections = []
    for i in range(len(extended_line_segments)):
        for j in range(i + 1, len(extended_line_segments)):
            intersection = line_intersection(
                extended_line_segments[i], extended_line_segments[j]
            )
            if intersection:
                line_intersections.append(intersection)
    polygon = Polygon(vertices)

    # Create example lines
    lines = [
        LineString([segment[:2], segment[2:]]) for segment in extended_line_segments
    ]
    polygon = Polygon(vertices)
    x, y = polygon.exterior.xy
    # fig, ax = plt.subplots()
    # ax.fill(x, y, alpha=0.5, fc='grey', label='Original Polygon')
    #
    # # Plot the lines that cut the polygon
    # for line in lines:
    #     x, y = line.xy
    #     ax.plot(x, y, label='Cutting Line')

    # Initialize a list to store the resulting polygons
    result_polygons = [polygon]

    # Cut the polygon using each line
    for line in lines:
        new_polygons = []
        for poly in result_polygons:
            if line.intersects(poly):
                # Extract the portion that lies within the polygon
                intersection = line.intersection(poly)
                if intersection.is_empty:
                    continue
                # Perform the actual splitting
                split_result = poly.difference(
                    Polygon(intersection.buffer(0.01).exterior)
                )
                if isinstance(split_result, MultiPolygon):
                    new_polygons.extend(list(split_result))
                elif isinstance(split_result, Polygon):
                    new_polygons.append(split_result)
            else:
                new_polygons.append(poly)
        result_polygons = new_polygons

    # Plot the resulting polygons in different colors
    # for poly in result_polygons:
    #     color = "#{:06x}".format(randint(0, 0xFFFFFF))
    #     x, y = poly.exterior.xy
    #     ax.fill(x, y, alpha=0.5, fc=color)
    return result_polygons


def get_regions(exp, t):
    im, skel_im = make_full_image(exp, t, downsizing=1000, dilation=5, edges=[])
    image = (im).astype(np.uint8)

    scale_unit = 1 / 1.725

    # Compute overlap
    def compute_overlap(params):
        x, y, angle = params
        polygon, R, t = create_polygon(x, y, angle, scale_unit)

        # Draw the polygon on a black image
        polygon_img = np.zeros_like(image)
        cv2.fillPoly(polygon_img, [polygon], 255)
        # Compute overlap between the image and the polygon
        overlap = np.sum((image / 255) * (polygon_img / 255))
        # Return negative overlap as we're using the minimize function
        return -overlap

    # Initial guess for the parameters
    deltas = np.array([20, 20, 30])
    init_params = [10, 30, 270]
    simplex = [init_params]
    for i in range(len(init_params)):
        new_point = np.copy(init_params)
        new_point[i] += deltas[i]
        simplex.append(new_point)

    # Convert to numpy array for use in scipy.minimize
    initial_simplex = np.array(simplex)
    # Bounds for the parameters ([x_min, x_max], [y_min, y_max], [angle_min, angle_max], [scale_min, scale_max])

    # Perform optimization to minimize overlap
    result = minimize(
        compute_overlap,
        init_params,
        method="Nelder-Mead",
        options={"initial_simplex": initial_simplex},
    )
    optimal_params = result.x

    # Draw the optimized polygon on the image
    optimal_polygon, angle, translation_vector = create_polygon(
        *optimal_params, scale_unit
    )
    # cv2.fillPoly(image, [optimal_polygon], 127)
    polygon_img = np.zeros_like(image)
    cv2.fillPoly(polygon_img, [optimal_polygon], 255)
    polygons = get_polygons(*optimal_params, scale_unit)

    centroids = [(polygon.centroid.y, polygon.centroid.x) for polygon in polygons]
    sorted_polygons = [
        polygon for centroid, polygon in sorted(zip(centroids, polygons))
    ]
    lists = [
        sorted_polygons[13:18],
        sorted_polygons[8:13],
        sorted_polygons[3:8],
        sorted_polygons[:3],
    ]
    final_sort = []
    for pre_sort in lists:
        centroids = [(polygon.centroid.x, polygon.centroid.y) for polygon in pre_sort]
        sorted_polygons_final = [
            polygon for centroid, polygon in sorted(zip(centroids, pre_sort))
        ]
        final_sort += sorted_polygons_final
    # fig, ax = plt.subplots()
    #
    # # Display the image with matplotlib
    # ax.imshow(image)
    # ax.imshow(polygon_img, cmap="Reds", alpha=0.5)
    # for i, polygon in enumerate(final_sort):
    #     # Extract the x and y coordinates
    #     x, y = polygon.exterior.xy
    #
    #     # Plot the polygon
    #     ax.fill(x, y, alpha=0.5, fc='b', label=f"Polygon {i}")
    #     ax.plot(x, y, 'r')
    #
    #     # Plot the index at the centroid
    #     centroid = polygon.centroid
    #     ax.text(centroid.x, centroid.y, str(i), fontsize=12, ha='center', va='center')
    return final_sort
