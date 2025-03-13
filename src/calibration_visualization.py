import numpy as np

def visualize_scene(K_cam, K_proj, RT_cam, RT_proj, point, projector_h_line_index):
    # Extract camera and projector parameters
    R_cam = RT_cam[:3, :3]
    t_cam = RT_cam[:3, 3]
    cam_origin_world = -np.linalg.inv(R_cam) @ t_cam  # Camera origin in world coordinates

    R_proj = RT_proj[:3, :3]
    t_proj = RT_proj[:3, 3]
    proj_origin_world = -np.linalg.inv(R_proj) @ t_proj  # Projector origin in world coordinates

    # Generate camera ray
    pixel_homogeneous = np.array([point[1], point[0], 1.0])
    ray_cam = np.linalg.inv(K_cam) @ pixel_homogeneous
    ray_world = np.linalg.inv(R_cam) @ ray_cam

    # Projector plane definition
    line_proj_homogeneous = np.array([0, projector_h_line_index, 1.0])
    point_on_line_proj = np.linalg.inv(K_proj) @ line_proj_homogeneous
    point_on_line_world = R_proj.T @ point_on_line_proj + t_proj

    normal_proj = np.linalg.inv(R_proj) @ np.array([0, 1, 0])

    # Ray-plane intersection
    numerator = np.dot(normal_proj, (point_on_line_world - cam_origin_world))
    denominator = np.dot(normal_proj, ray_world)
    t = numerator / denominator
    intersection_world = cam_origin_world + t * ray_world

    # Visualization settings
    axis_length = 10
    num_points = 50  # Number of points to create each axis

    def create_axis(origin, direction, length, num_points):
        return np.array([origin + direction * (i / num_points) * length for i in range(num_points + 1)])

    # Camera axes
    cam_x = create_axis(cam_origin_world, R_cam[:, 0], axis_length, num_points)  # X-axis
    cam_y = create_axis(cam_origin_world, R_cam[:, 1], axis_length, num_points)  # Y-axis
    cam_z = create_axis(cam_origin_world, R_cam[:, 2], axis_length, num_points)  # Z-axis

    # Projector axes
    proj_x = create_axis(proj_origin_world, R_proj[:, 0], axis_length, num_points)
    proj_y = create_axis(proj_origin_world, R_proj[:, 1], axis_length, num_points)
    proj_z = create_axis(proj_origin_world, R_proj[:, 2], axis_length, num_points)

    # World axes
    world_x = create_axis(np.zeros(3), np.array([1, 0, 0]), axis_length, num_points)
    world_y = create_axis(np.zeros(3), np.array([0, 1, 0]), axis_length, num_points)
    world_z = create_axis(np.zeros(3), np.array([0, 0, 1]), axis_length, num_points)

    # Ray visualization
    ray_points = np.vstack([
        cam_origin_world,
        intersection_world
    ])

    # Plane visualization (generate a grid)
    plane_size = 20
    plane_points = []
    z_plane_points = []

    for x in np.linspace(-plane_size, plane_size, 20):
        for z in np.linspace(-plane_size, plane_size, 20):
            point_on_plane = point_on_line_world + x * R_proj[:, 0] + z * R_proj[:, 2]
            plane_points.append(point_on_plane)
            z_plane_points.append([x, z, 0])  # z=0 plane

    plane_points = np.array(plane_points)
    z_plane_points = np.array(z_plane_points)

    # Save to XYZ file
    with open('visualization.xyz', 'w') as f:
        def write_points(points, color):
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]} {color[0]} {color[1]} {color[2]}\n")

        # Camera axes
        write_points(cam_x, (255, 0, 0))  # Red for X
        write_points(cam_y, (0, 255, 0))  # Green for Y
        write_points(cam_z, (0, 0, 255))  # Blue for Z

        # Projector axes
        write_points(proj_x, (255, 0, 0))
        write_points(proj_y, (0, 255, 0))
        write_points(proj_z, (0, 0, 255))

        # World axes
        write_points(world_x, (255, 0, 0))
        write_points(world_y, (0, 255, 0))
        write_points(world_z, (0, 0, 255))

        # # Ray points (Yellow)
        # write_points(ray_points, (255, 255, 0))

        # # Plane points (Gray)
        # write_points(plane_points, (200, 200, 200))

        # z=0 Plane (Darker Gray)
        write_points(z_plane_points, (150, 150, 150))

    print("3D visualization saved as visualization.xyz")

K_proj = np.load("camera_projector_parameter/K_proj.npy")
K_cam = np.load("camera_projector_parameter/K_cam.npy")
RT_proj = np.load("camera_projector_parameter/RT_proj.npy")
RT_cam = np.load("camera_projector_parameter/RT_cam.npy")

projector_h_line_index = 50
point = np.array([0.8159329640266102, 4])
visualize_scene(K_cam, K_proj, RT_cam, RT_proj, point, projector_h_line_index)