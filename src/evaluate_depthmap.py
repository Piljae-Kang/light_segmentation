import open3d as o3d
import numpy as np
import os

def create_mesh_poisson(pcd,folder_path, depth_range):

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

    for octree_depth in range(depth_range[0], depth_range[1]+1):

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=octree_depth)

        gray_color = np.full((np.asarray(mesh.vertices).shape[0], 3), 150 / 255.0)
        mesh.vertex_colors = o3d.utility.Vector3dVector(gray_color)

        o3d.io.write_triangle_mesh(f"{folder_path}/mesh_depth_{octree_depth}.ply", mesh)
        print(f"{octree_depth}th mesh is completed")




if __name__ == "__main__":

    pcd_path = "/media/piljae/X31/result/original_hubitz_pattern/metal_shaft/1/com0.ply"
    #pcd_path = "/media/piljae/X31/Dataset/Hubitz/direct_pattern_experiment_result/normalize/median3x3/alpha29/com0.ply"

    pcd = o3d.io.read_point_cloud(pcd_path)
    folder_path = "/".join(pcd_path.split("/")[:-1])

    #create_mesh_poisson(pcd,folder_path, depth_range=(5,8))

    depth = 6
    mesh = o3d.io.read_triangle_mesh(f"{folder_path}/mesh_depth_{depth}.ply")

    mesh_kd_tree = o3d.geometry.KDTreeFlann(mesh)

    distances = []

    for point in pcd.points:

        [_, idx, _] = mesh_kd_tree.search_knn_vector_3d(point , 1)
        closest_point = np.asarray(mesh.vertices)[idx[0]]

        distance = np.linalg.norm(point - closest_point)
        distances.append(distance)

    print(np.mean(distances))
    print(np.sum(np.power(distances - np.mean(distances), 2)/len(distances)))





    #pcd_path = "/media/piljae/X31/result/original_hubitz_pattern/metal_shaft/1/com0.ply"
    pcd_path = "/media/piljae/X31/Dataset/Hubitz/direct_pattern_experiment_result/original/median3x3/alpha29/com0.ply"

    pcd = o3d.io.read_point_cloud(pcd_path)
    folder_path = "/".join(pcd_path.split("/")[:-1])

    #create_mesh_poisson(pcd,folder_path, depth_range=(5,8))

    depth = 6
    mesh = o3d.io.read_triangle_mesh(f"{folder_path}/mesh_depth_{depth}.ply")

    mesh_kd_tree = o3d.geometry.KDTreeFlann(mesh)

    distances2 = []

    for point in pcd.points:

        [_, idx, _] = mesh_kd_tree.search_knn_vector_3d(point , 1)
        closest_point = np.asarray(mesh.vertices)[idx[0]]

        distance = np.linalg.norm(point - closest_point)
        distances2.append(distance)

    print(np.mean(distances2))
    print(np.sum(np.power(distances2 - np.mean(distances2), 2)/len(distances2)))

    print(len(distances2))

    import matplotlib.pyplot as plt

    # 히스토그램 생성
    plt.hist(distances, bins=300, alpha=0.5, label='Dataset 1', color='blue', edgecolor='black', density=True)
    plt.hist(distances2, bins=300, alpha=0.5, label='Dataset 2', color='red', edgecolor='black', density=True)

    plt.xlim(0.0, 0.5)
    plt.title('Histogram of Distances')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()