# spatchcocking.py
"""
Spatchcocking: Chicken Neural Tube Data Processing Utility Module

This module provides a geometric pipeline for transforming 3D neural tube lumen 
data into flattened 2D projections. It handles the transition from volumetric 
masks to manifold meshes, extracts topological centerline (medial axes), 
and performs the "spatchcock" unfolding transformation.

NOTE: For jupyter notebook you have to set backend
    ```
    from vedo import settings
    settings.default_backend = "vtk"
    ```

The pipeline generally follows this order:
    1. Mask to Mesh (STL)
    2. Medial Axis/Spline Extraction
    3. Curvature & Signal Mapping (e.g., PHH3)
    4. 3D to 2D Flattening
    
Main Features:
    - Mesh Generation: Converts binary TIFF lumen masks to STL meshes.
    - Differential Geometry: Calculates surface curvature and topological metrics.
    - Signal Extraction: Maps PHH3 (mitotic) signals onto the lumen surface.
    - Flattening: Unrolls the 3D geometry along a neural tube axis for 2D analysis.

Example:
    >>> from spatchcocking import *
    >>> import spatchcocking as sp
    
Dependencies:
    - vedo: For visualization and maniputating mesh data.
    - tifffile: Loading volumetric biological data.
    - scipy: Spline interpolation and medial axis extraction.
    
"""

# import your mesh/geometry libraries here

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage import measure, morphology, filters
import trimesh
import tifffile
from scipy import ndimage
import os
from vedo import *

# =============================================================================
# 1. MESH GENERATION (MASK TO STL)
# =============================================================================


def load_tiff_stack(file_path):
    """
    Load 3D TIFF stack (Mask of lumen)
    Note: Lumen = 1, Background = 0
    
    Parameters:
    file_path (str): Path to TIFF stack file
    
    Returns:
    numpy.ndarray: 3D array of the stack
    """
    try:
        # Load TIFF stack
        stack = tifffile.imread(file_path)
        
        # Handle different TIFF formats
        if stack.ndim == 4:
            # Multi-channel, take first channel
            stack = stack[0] if stack.shape[0] < stack.shape[-1] else stack[:, :, :, 0]
        elif stack.ndim == 2:
            # Single slice, add dimension
            stack = stack[np.newaxis, :, :]
        
        print(f"Loaded TIFF stack with shape: {stack.shape}")
        print(f"Data type: {stack.dtype}")
        print(f"Value range: [{stack.min()}, {stack.max()}]")
        
        return stack
        
    except Exception as e:
        print(f"Error loading TIFF file: {e}")
        return None

def preprocess_mask(mask, 
                   threshold=0.5, 
                   remove_small_objects=True,
                   min_size=100,
                   smooth=True,
                   fill_holes=True):
    """
    Preprocess the 3D mask before mesh extraction
    
    Parameters:
    mask (numpy.ndarray): 3D mask array
    threshold (float): Threshold for binarization
    remove_small_objects (bool): Remove small connected components
    min_size (int): Minimum size for connected components
    smooth (bool): Apply smoothing
    fill_holes (bool): Fill holes in the mask
    
    Returns:
    numpy.ndarray: Preprocessed binary mask
    """
    print("Preprocessing mask...")
    
    # Convert to binary if not already
    if mask.dtype != bool:
        binary_mask = mask > threshold * mask.max()
    else:
        binary_mask = mask.copy()
    
    print(f"Binary mask has {np.sum(binary_mask)} voxels")
    
    # Remove small objects
    if remove_small_objects:
        print("Removing small objects...")
        binary_mask = morphology.remove_small_objects(binary_mask, min_size=min_size)
        print(f"After cleaning: {np.sum(binary_mask)} voxels")
    
    # Fill holes
    if fill_holes:
        print("Filling holes...")
        binary_mask = ndimage.binary_fill_holes(binary_mask)
        print(f"After filling holes: {np.sum(binary_mask)} voxels")
    
    # Smooth the mask
    if smooth:
        print("Smoothing mask...")
        # Apply slight gaussian smoothing
        smoothed = filters.gaussian(binary_mask.astype(float), sigma=0.5)
        binary_mask = smoothed > 0.5
        print(f"After smoothing: {np.sum(binary_mask)} voxels")
    
    return binary_mask

def extract_mesh_marching_cubes(mask, spacing=(1.0, 1.0, 1.0), level=0.5):
    """
    Extract mesh using marching cubes algorithm
    
    Parameters:
    mask (numpy.ndarray): 3D binary mask
    spacing (tuple): Voxel spacing (z, y, x)
    level (float): Iso-surface level
    
    Returns:
    tuple: (vertices, faces, normals, values)
    """
    print("Extracting mesh using marching cubes...")
    
    # Run marching cubes
    vertices, faces, normals, values = measure.marching_cubes(
        mask.astype(float), 
        level=level, 
        spacing=spacing,
        method='lewiner'  # More robust than 'classic'
    )
    
    print(f"Marching cubes result:")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Faces: {len(faces)}")
    print(f"  Normals: {len(normals)}")
    
    return vertices, faces, normals, values

def extract_mesh_dual_contouring(mask, spacing=(1.0, 1.0, 1.0)):
    """
    Extract mesh using dual contouring (via trimesh)
    
    Parameters:
    mask (numpy.ndarray): 3D binary mask
    spacing (tuple): Voxel spacing (z, y, x)
    
    Returns:
    tuple: (vertices, faces)
    """
    print("Extracting mesh using dual contouring...")
    
    try:
        # Create voxel grid
        voxel_grid = trimesh.voxel.VoxelGrid(mask, spacing=spacing)
        
        # Extract mesh using dual contouring
        mesh = voxel_grid.marching_cubes
        
        if mesh is None:
            print("Dual contouring failed, trying alternative method...")
            # Alternative: create mesh from voxel centers
            mesh = voxel_grid.as_boxes()
        
        vertices = mesh.vertices
        faces = mesh.faces
        
        print(f"Dual contouring result:")
        print(f"  Vertices: {len(vertices)}")
        print(f"  Faces: {len(faces)}")
        
        return vertices, faces
        
    except Exception as e:
        print(f"Dual contouring failed: {e}")
        return None, None

def save_mesh(vertices, faces, output_path, format='ply'):
    """
    Save mesh to file
    
    Parameters:
    vertices (numpy.ndarray): Mesh vertices
    faces (numpy.ndarray): Mesh faces
    output_path (str): Output file path
    format (str): Output format ('ply', 'obj', 'stl')
    """
    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save mesh
    mesh.export(output_path, file_type=format)
    print(f"Mesh saved to: {output_path}")

def tiff_stack_to_mesh(tiff_path=None,
                      output_path=None,
                      spacing=(1.0, 1.0, 1.0),
                      method='marching_cubes',
                      preprocess=True,
                      ):
    """
    Complete pipeline to convert 3D TIFF stack to mesh
    
    Parameters:
    tiff_path (str): Path to TIFF stack file 
    output_path (str): Output mesh file path
    spacing (tuple): Voxel spacing (z, y, x)
    method (str): Extraction method ('marching_cubes', 'dual_contouring')
    preprocess (bool): Apply preprocessing
    simplify (bool): Simplify mesh
    smooth (bool): Smooth mesh
    visualize (bool): Show visualization
    
    Returns:
    tuple: (vertices, faces) of the extracted mesh
    """
    
    # Step 1: Load TIFF stack
    if tiff_path is None:
        print("put correct tiff path...")
        return None, None
    else:
        mask = load_tiff_stack(tiff_path)
        if mask is None:
            return None, None
    
    # Step 2: Preprocess mask
    if preprocess:
        mask = preprocess_mask(mask)
    
    # Step 3: Extract mesh
    if method == 'marching_cubes':
        vertices, faces, normals, values = extract_mesh_marching_cubes(mask, spacing)
    elif method == 'dual_contouring':
        vertices, faces = extract_mesh_dual_contouring(mask, spacing)
        if vertices is None:
            print("Dual contouring failed, falling back to marching cubes...")
            vertices, faces, normals, values = extract_mesh_marching_cubes(mask, spacing)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    
    # Step 4: Save mesh
    if output_path:
        save_mesh(vertices, faces, output_path)
    
    return vertices, faces

def strippathname(fullpath):
    """
    Strip path and return only the filename without extension.
    
    Parameters:
    fullpath (str): Full file path
    
    Returns:
    str: Filename without extension
    """
    base = os.path.basename(fullpath)
    return os.path.splitext(base)[0]

## IMPORTANT ONE IS HERE
def get_mesh(mask_path, 
             px2umz, px2umxy,
             check=None):
    """
    Converts a 3D binary TIFF mask into an STL mesh.    
    
    Args:
        mask_path (str): Path to TIFF stack file 
        px2umz (float): z step size in micron
        px2umxy (float): xy pixel length in micron
        check (bool, optional): If True, opens a Vedo Plotter window to compare 
            the raw mesh vs. the processed mesh. Defaults to None (No plotting).
        
    Returns:
        mesh1 (mesh_object): vedo mesh object            
    """
    namefile = strippathname(mask_path)

    # Extract vertices and faces from the TIFF volume
    vertices, faces = tiff_stack_to_mesh(
        tiff_path=mask_path,  
        output_path= namefile + '.ply',
        spacing=(px2umz, px2umxy, px2umxy),  # Adjust spacing based on your data
        method= 'marching_cubes',  # or 'dual_contouring'
        preprocess=True,
    )
    # Smoothing a mesh, use vedo to smooth and save the mesh
    mesh0 = Mesh(namefile + '.ply').lw(1)

    mesh1 = mesh0.clone()
    mesh1.decimate(n=1000).subdivide(3).smooth().compute_normals()
    # mesh1.decimate(n=200).subdivide(3).smooth().compute_normals()
    mesh1.c('light blue').lw(0).lighting('glossy').phong()
    # other useful filters to combine are
    # mesh.decimate(), clean(), smooth()


    if check:
        plt = Plotter(shape=(1, 2), axes=7)
        plt.at(0).show(mesh0.wireframe()) # "Raw Mesh" 
        plt.at(1).show(mesh1.wireframe()) # "Processed Mesh"
        doc = Text2D("Comparing the raw vs Proceesed mesh")
        plt += doc
        plt.interactive().close()
    
    # Export and Return
    mesh1.write(namefile + '-mesh.stl')
    print("Saved:", namefile + '-mesh.stl')
    return mesh1

# =============================================================================
# 2. CURVATURE CALCULATION
# =============================================================================

def getDefaultname(namefile):
    """
    Return a default name derived from files in the current directory when no name is provided.

    Args:
        namefile (str or None):
            If a string is provided, it is returned unchanged. If None, the function
            searches the current working directory for files ending with the suffix
            '.tif' and returns the filename with that suffix removed.

    Returns:
        namefile (str)
    """
    if namefile==None:
        names = [f[:-4] for f in os.listdir('.') if f.endswith('.tif')]
        if names:
            namefile = names[0]
        else:
            # Case: No tif files found
            namefile = f"noname"
    return namefile

def getTightercmap(values, sigma=3):
    """
    For highlighting contrast
    
    Args:
        values (np.array):
        sigma (float): to clip the standard deviation
    
    Returns:
        valuemin, valuemax (float)

    """
    valuemin = np.median(values) -sigma*np.std(values)
    valuemax = np.median(values) +sigma*np.std(values)
    return valuemin, valuemax

## IMPORTANT ONE IS HERE
def getProperCurvature(msh, depth, 
                       namefile=None,
                       type="Gaussian",
                       check=None):
    """
    Get Curvature by fitting Quadratic surface on the a local neighborhood.
    
    Args:
        msh (mesh_object): Mesh
        depth (int): Count of adjacent points to fit the quadratic surface
        namefile (str): Name to store the curvature values
        type (str): Type of curvature to store/return. 
            Options: "Gaussian" or "Mean".
        check (bool, optional): If True, opens a Vedo Plotter window to 
            show the curvature. Defaults to None (No plotting).
        
    Returns:
        msh1 (mesh_object): Mesh with Curvature
        saves curvature values in a curvature.npy file
    
    Thanks Marco Musy. https://github.com/marcomusy/vedo/discussions/1272
    """
    msh1 = msh.clone()
    msh1.compute_normals()

    # Compute adjacency list for the mesh vertices
    adlist = msh1.compute_adjacency()

    # Compute curvature at all points by fitting a quadratic surface

    curvs_g = []
    curvs_m = []
    pts1 = msh1.points
    for i in progressbar(range(msh1.npoints)):
        ids = msh1.find_adjacent_vertices(i, depth=depth, adjacency_list=adlist)
        bpts = msh1.points[ids]
        _, res = project_point_on_variety(pts1[i], bpts, degree=2, compute_curvature=True)
        curvs_g.append(res[4])
        curvs_m.append(res[5])
    

    # Use () to call the method
    if type.lower() == 'gaussian':
        msh1.pointdata["Gauss_Curvature"] = curvs_g
        curvs_val = msh1.pointdata["Gauss_Curvature"]
    elif type.lower() == 'mean':
        msh1.pointdata["Mean_Curvature"] = curvs_m
        curvs_val = msh1.pointdata["Mean_Curvature"]
    else:
        # Optional: Fallback or error if someone types something else
        raise ValueError(f"Unknown curvature type: {type}. Choose 'Gaussian' or 'Mean'.")
    
    namefile = getDefaultname(namefile)
    
    meshpts = msh1.vertices
    curv_col = curvs_val.reshape(-1, 1) 
    combinedcurvaturepos = np.hstack((meshpts, curv_col))
    np.save(namefile + '-curvature.npy', combinedcurvaturepos) # save the files
    
    if check:
        vmin, vmax = getTightercmap(curvs_val) # just to visualize the curvatures
        __doc__ = "Curvature Distribution"
        show(msh1.cmap('RdBu',curvs_val,vmin=vmin, vmax=vmax).add_scalarbar(),  __doc__)   

        
    return msh1

# =============================================================================
# 3. MEDIAL AXIS SPLINE
# =============================================================================

def selectPointsonMesh(mesh,
                       namefile=None):
    """
    Select points on the dorsal surface of the tube. starting from posterior end

    Args:
        mesh (mesh_object): mesh
        namefile (str): label for saved file

    Returns:
        selected_points (array): selected points on the dorsal surface
    """


    # mesh = mesh.alpha(0.5)
    mesh.pickable(True)

    # Variables to store selected points and their marker objects
    selected_points = []
    selected_markers = []

    # Create the plotter
    plt = Plotter(title="Click to select points. Press 'c' to clear.", axes=9)

    # Callback function for clicking
    def on_left_click(evt):
        if evt.actor and evt.picked3d is not None:
            point_coords = evt.picked3d
            print("Selected point:", point_coords)

            # Store point and add a red marker
            selected_points.append(point_coords)
            marker = Sphere(pos=point_coords, r=5, c="red")
            selected_markers.append(marker)
            plt.add(marker)  # Use plt here, not plotter
            plt.render()

    # Keypress callback: clear markers
    def on_keypress(evt):
        if evt.keypress  == "c":
            print("Clearing all selected points...")
            for marker in selected_markers:
                plt.remove(marker)
            selected_markers.clear()
            selected_points.clear()
            plt.render()
            
            
    # Add callbacks
    plt.add_callback("left_click", on_left_click)
    plt.add_callback("key_press", on_keypress)


    # Show mesh and interact
    doc = Text2D("Select points on the dorsal surface of the tube.\n Starting from posterior end \n In that order \n Press 'c' to clear")
    plt += doc
    plt.show(mesh, interactive=True).close()
    
    namefile = getDefaultname(namefile)

    np.save(namefile + "-endpts.npy", selected_points)

    return selected_points

## IMPORTANT ONE IS HERE
def getAxis(mesh, 
            endpts,
            namefile=None,
            num_points = 12,
            N=15,
            check=None):
    """
    do the mls smoothing iterations, for getting the skeleton axis
    
    Args:
        msh (mesh_object): Mesh
        endpts (array): end points arrays
        namefile (str): Name to store the axis points
        num_points (int): number of final axis points
        N (int): Number of iterations for smoothening (10-30)
        check (bool, optional): If True, opens a Vedo Plotter window to 
            show the spline. Defaults to None (No plotting).
    Returns:
        decimated_points (array): decimated axis points 
    """
    
    pcl = Points(mesh.vertices).subsample(0.02)

    plt = Plotter(N=N, axes=1)
    for i in range(N):
        pcl = pcl.clone().smooth_mls_1d(f=1).color(i) 
        # f goes from 0-2, 0.5 works well. 1 if you get strange curves
        plt.at(i).show(f"iteration {i+1}", pcl, elevation=-8)
    plt.interactive().close()

    pcl.subsample(0.1)

    # show(pcl.color('k'),mesh, axes=1)

    ## closest point to the selected point, because we need to know where are the pts
    idx = pcl.closest_point(endpts[0], return_point_id=True)
    # print("closest point index:", idx) 

    ## get the points and then order them and connect them with a spline
    pclpts = pcl.vertices
    # print(pclpts)   

    # points are randomly is ordered, we have to order them posterior to anterior
    order = nearest_neighbor_order(pclpts, start=idx)
    ordered_pts = pclpts[order]

    # ISSUE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # add the end points to the medial axis (Lets keep it as it is for now)
    # ordered_pts = np.vstack([endpts[0], ordered_pts, endpts[1]])

    ## add two extra points at the ends, one in the direction of the first segment
    ## one in the direction of the last segment
    
    first_dir = ordered_pts[1] - ordered_pts[0]
    last_dir = ordered_pts[-1] - ordered_pts[-2]    

    delta = 2
    new_start = ordered_pts[0] - first_dir*delta
    new_end = ordered_pts[-1] + last_dir*delta

    new_ordered_pts = np.vstack([new_start, ordered_pts, new_end])
    pspline = Spline(new_ordered_pts, closed=False).c('green')

    psplinepts = pspline.vertices
    
    # Calculate the step size to get exactly num_points
    indices = np.linspace(0, len(psplinepts) - 1, num_points).astype(int)

    decimated_points = psplinepts[indices]
    
    if check:
        ptsd = Points(decimated_points).c('red').ps(8)
        new_pspline1 = Spline(decimated_points, closed=False).c('green')
        show(new_pspline1, ptsd, mesh.alpha(0.5), axes=1)

    
    # save the new pspline pts to numpy file
    namefile = getDefaultname(namefile)
    np.save(namefile + "-axis.npy", decimated_points)
    
    return decimated_points

def getnormal(pt1, pt2):
    """
    Get normal between two points
    
    Args:
        pt1 (3d nparray): start pt
        pt2 (3d nparray): end pt
        
    Returns:
        normalized_projection_axis
    """    
    # pt1 is start, pt2 is end
    projection_axis = pt2- pt1
    # Calculate the norm of the resulting vector
    norm = np.linalg.norm(projection_axis)
    # Divide by the norm
    normalized_projection_axis = projection_axis / norm
    return normalized_projection_axis

# =============================================================================
# 4. STRAIGTHENING THE MESH
# =============================================================================

def find_closest_dorsal_points(axispts, dorsalpts):
    """
    Finds the index of the closest dorsalpt for every axispt.
    
    Args:
        axispts (np.array): Shape (N, 3)
        dorsalpts (np.array): Shape (M, 3)
        
    Returns:
        closest_indices (np.array): Indices of dorsalpts closest to each axispt.
        closest_points (np.array): The actual coordinates from dorsalpts.
    """
    
    # Use broadcasting to calculate the difference between every pair of points
    # axispts[:, np.newaxis, :] shape: (N, 1, 3)
    # dorsalpts[np.newaxis, :, :] shape: (1, M, 3)
    # Difference shape: (N, M, 3)
    diff = axispts[:, np.newaxis, :] - dorsalpts[np.newaxis, :, :]
    
    # Calculate squared Euclidean distance: (N, M)
    # We sum along the last axis (the coordinates)
    dist_sq = np.sum(diff**2, axis=2)
    
    # For each axispt (rows), find the index of the minimum distance in dorsalpts (cols)
    closest_indices = np.argmin(dist_sq, axis=1)
    
    # Map indices back to the actual coordinates
    closest_points = dorsalpts[closest_indices]
    
    return closest_indices, closest_points

def getPlanes(mesh, axispts, endpts,
                  skip_index = np.array([]),
                  check=None):
    """
    Get planes along the axis
    
    Args:
        mesh (mesh_object): Mesh
        axispts (array): axis points arrays
        skip_index (array): number of planes which look wrong
        check (bool):  to see the details of process
        
    Returns:
        axis_info = {
        'axis': axispts,
        'axis normals': normals,
        'dorsal normals': dnormals
        }
    """
    axis = Points(axispts).c('red').ps(8)


    # Compute Tangents (Direction of the path)
    # This returns the unit vectors pointing along the curve
    normals = np.gradient(axispts, axis=0)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    
    ## Points closer to dorsal line    
    dorsal_line = Spline(endpts, 100, closed=False)
    dorsal_pts = dorsal_line.vertices
    indices, subset_dorsal_pts = find_closest_dorsal_points(axispts, dorsal_pts)
    
    # Vector from axispts to dorsal_pts (v = P - O)
    v = subset_dorsal_pts - axispts

    # Calculate the projection of v onto the plane
    # dist is (v dot n). We subtract the component parallel to the normal.
    dist = np.sum(v * normals, axis=1, keepdims=True)
    dnormal_vec = v - (dist * normals)
    propt = dnormal_vec + axispts # projected point. we dont need this
    
    # Normalize the resulting vector to get unit dnormals
    dnormals = dnormal_vec / np.linalg.norm(dnormal_vec, axis=1, keepdims=True)


    # Initialize an empty list to hold flag posts
    fss = []
    for idx, p in enumerate(axispts):
        # Create a flag post at point p with the index as the label
        fs = axis.flagpost(f"{idx}", p, c='orange',vspacing = 0.1 )
        fss.append(fs)

    if check:
        plt = Plotter(axes=1)
        plt += mesh.alpha(0.3)
        plt += axis
        plt += fss
        # plt += Points(subset_dorsal_pts, r=20, c='red')
        # plt += Points(propt, r=20, c='blue')


    for i in range(0, len(axispts)):
        if i in skip_index:
            continue
        normal = normals[i, :]
        othernormal = dnormals[i,:]
        pt1 = axispts[i,:]
        plane = Plane(pos=pt1, normal=normal, s=(1000, 1000)).alpha(0.1).c('green')

        if check:
            plt += Arrow(pt1, pt1+normal*50)
            plt += Arrow(pt1, pt1+othernormal*50)
            plt += plane

    
    if check:
        doc = Text2D("REMEMBER INTERSECTING PLANES \n Planes along the axis points, \n see if the planes intersect, \n also plane size = 1000, \n adjust dist threshold if planes are too small")
        plt += doc
        plt.show().close() 
    
    axis_info = {
    'axis': axispts,
    'axis normals': normals,
    'dorsal normals': dnormals
    }
    
    return axis_info

def nearest_neighbor_order(points, start=0):
    """
    Compute a visit order of 3D (or n-D) points using a greedy nearest-neighbor heuristic.

    Parameters
    - points: array-like of shape (N, D). Sequence of N points in D-dimensional space.
    - start: int, optional. Index of the starting point in `points`. Default 0.

    Returns
    - order: list of int, length N. Indices into `points` giving the visitation order,
    where consecutive indices are chosen greedily as the closest unvisited point.

    Notes
    - This is a greedy heuristic for constructing a single polyline through all points.
    - It does NOT guarantee the globally shortest path (the Travelling Salesman Problem).
    - Time complexity: O(N^2 * D) in this implementation (each step searches remaining points).
    - Space complexity: O(N) for bookkeeping.
    - For large N, consider spatial acceleration (KDTree) or heuristic improvement (2-opt).
    """

    # Convert input to a NumPy array to ensure vectorized numeric operations.
    # If `points` is already an ndarray, this is cheap (returns view); otherwise it creates one.
    pts = np.asarray(points)

    # N is the number of points (rows). This determines how many iterations we will do.
    N = len(pts)

    # `remaining` holds indices of points not yet visited.
    # Using a set gives O(1) average-time membership and removal.
    remaining = set(range(N))

    # `order` stores the visiting sequence of indices. Start with the given index.
    order = [start]

    # Remove the start index from the set of remaining (it's now visited).
    remaining.remove(start)

    # `cur` is the index of the current point (we start there).
    cur = start

    # Loop until all points have been visited.
    while remaining:
        # `cur_pt` is the coordinates of the current point.
        cur_pt = pts[cur]

        # Find the index in `remaining` whose point has the minimum squared Euclidean distance
        # to `cur_pt`. We use squared distance (sum of squared differences) because square
        # root is monotonic and unnecessary for comparisons; this is slightly faster.
        nearest = min(remaining, key=lambda i: np.sum((cur_pt - pts[i])**2))

        # Append the found index to the order (we will visit it next).
        order.append(nearest)

        # Mark it as visited by removing from `remaining`.
        remaining.remove(nearest)

        # Update current index to the newly visited point and repeat.
        cur = nearest

    # Return the full visiting order (list of indices).
    return order

def transform_to_radial(points, center, normal, dnormal):
    
    """Transform 3D points to radial coordinates around a center point and normal vector.

        Args:
            points: array-like of shape (N, 3). Sequence of N 3D points.
            center: array-like of shape (3,). The center point for the radial transformation.
            normal: array-like of shape (3,). The normal vector defining the axis of rotation.
            dnormal: array-like of shape (3,). The normal vector defining the axis of rotation in dorsal direction.

        Returns
            radial_coords: array of shape (N, 3). Each row contains (radius, angle, height) for each point. where radius is the distance from the axis, angle is the angle around the axis in radians, and height is the projection along the normal vector.
    """
    
    # Ensure inputs are arrays
    pts = np.asarray(points)
    n = normal / np.linalg.norm(normal)
    y_axis = dnormal / np.linalg.norm(dnormal)
    x_axis = np.cross(y_axis, n) # Strictly orthogonal X
    x_axis /= np.linalg.norm(x_axis)

    # Vector from center to points
    vec = pts - center
    
    # Projection and math (Vectorized)
    height = np.dot(vec, n)
    # Reshape height for broadcasting: (N, 1)
    proj = vec - height[:, np.newaxis] * n
    radius = np.linalg.norm(proj, axis=1)
    
    # Angle relative to our specific x_axis and y_axis
    angle = np.arctan2(np.dot(proj, y_axis), np.dot(proj, x_axis))
    
    # Stack into (N, 3)
    return np.column_stack((radius, angle, height))

def transform_to_cartesian(radial_coords, center, normal, dnormal):
    """Transform 3D cartesian points from radial coordinates around a center point and normal vector.

        Args:
            points: array-like of shape (N, 3). Sequence of N 3D points.
            center: array-like of shape (3,). The center point for the radial transformation.
            normal: array-like of shape (3,). The normal vector defining the axis of rotation.
            dnormal: array-like of shape (3,). The normal vector defining the axis of rotation in dorsal direction.

        Returns
           points : array of shape (N, 3). cartesian coordinates.
    """
    radial_coords = np.asarray(radial_coords)
    radius = radial_coords[:, 0]
    angle = radial_coords[:, 1]
    height = radial_coords[:, 2]

    # Reconstruct the EXACT same coordinate basis logic
    n = normal / np.linalg.norm(normal)
    y_axis = dnormal / np.linalg.norm(dnormal)
    x_axis = np.cross(y_axis, n)
    x_axis /= np.linalg.norm(x_axis)

    # Vectorized reconstruction
    # (N, 1) * (3,) results in (N, 3)
    proj = (radius * np.cos(angle))[:, np.newaxis] * x_axis + (radius * np.sin(angle))[:, np.newaxis] * y_axis
           
    points = center + proj + (height[:, np.newaxis] * n)
    
    return points

def query4skip_index():
    # Ask for input for planes to be skiped
    user_input = input("Enter a list of indexes to be skipped separated by spaces (or press Enter for empty): ")

    try:
        # .split() on an empty string "" returns []
        # np.array([]) creates an empty array
        items = user_input.split()
        
        if not items:
            skip_index = np.array([], dtype=int)
            print("Input was empty. Stored an empty NumPy array.")
        else:
            skip_index = np.array(items, dtype=int)
            print("skip index Array:", skip_index)
            
    except ValueError:
        print("Error: Please ensure you only enter integers.")
    return skip_index

## IMPORTANT ONE IS HERE
def getDeformedmesh(mesh, axis_info,
                    namefile=None,
                    skip_index =np.array([]),
                    dists_threshold=300,
                    check=None):
    """
    Straightening the mesh
    
    Args:
        mesh (mesh_object): Mesh
        axis_info (dictionary): axis points arrays and normals
            axis_info = {
                'axis': axispts,
                'axis normals': normals,
                'dorsal normals': dnormals
                }
        skip_index (array): number of planes which look wrong
        namefile (str): Name to store the curvature values
        dists_threshold (int): distance to exclude points when mesh is very curved
        check (bool): to see the details of process
    
    Returns:
        deformed_mesh (mesh_object): Deformed mesh
    """
    normals = axis_info['axis normals']
    dnormals = axis_info['dorsal normals']
    axispts = axis_info['axis']
    
    
    namefile = getDefaultname(namefile)    # get name
    curvaturevals = np.load(namefile+"-curvature.npy") # load curvature
        
    mesh.c('lightblue').decimate(n=1000) # its important to decimate

    # load mesh pts and assign curvature
    meshpts = Points(curvaturevals[:,:3]).ps(8) 
    meshpts.pointdata["curvature"] = curvaturevals[:,3]

    # apply pt data on the mesh
    mesh.interpolate_data_from(meshpts, n=2)

    curvature0 = mesh.pointdata['curvature']

    vmin, vmax = getTightercmap(curvaturevals[:,3], sigma=1) # just to visualize the curvatures
    mesh.cmap('RdBu',mesh.pointdata['curvature'],vmin=vmin, vmax=vmax).add_scalarbar()


    # get new normal from first two axis points
    newNormal = np.array([0, 0, 1]) #normals[0]
    newdnormal = np.array([0, 1, 0])#dnormals[0]
    # %% straighten the axis points
    # start from the first point and add distance along newNormal
    straightened_pts = [axispts[0]]
    for i in range(1, len(axispts)):
        dist = np.linalg.norm(axispts[i] - axispts[i-1])
        new_pt = straightened_pts[-1] + newNormal * dist
        straightened_pts.append(new_pt) 
    straightened_pts = np.array(straightened_pts)   

    if check:
        # convert to vedo Points for visualization
        axis = Points(axispts).c('red').ps(8)
        straightened_axis = Points(straightened_pts).c('blue').ps(8)
        plt = Plotter(axes=1)
        plt += mesh.alpha(0.3)
        plt += axis
        plt += straightened_axis

    # empty list of points in orginal mesh corresponding to each axis point and their transformed positions
    original_points = []
    transformed_points = []

    for i in range(0, len(axispts)):
    
        # if i is in skip_index, skip this iteration
        if i in skip_index:
            continue

        pt1 = axispts[i]
        normal = normals[i]
        dnormal = dnormals[i]

            
        cut = mesh.clone().intersect_with_plane(pt1, normal) 
        # cut the mesh with the plane at pt1 with normal
        propoints = cut.vertices # get the points of the cut mesh

        # if no points found, skip
        if len(propoints) == 0:
            # skip this iteration
            continue

        # check if there are two clusters of points in the propoints, by checking their distance from pt1 and then plotting a histogram
        dists = np.linalg.norm(propoints - pt1, axis=1)
        propoints = propoints[dists < dists_threshold]     # delete points that are too far away
        original_points.append(propoints)     # append original points

        # transform the propoints to radial cordinations around pt1 and normal
        radial_coords = transform_to_radial(propoints, pt1, normal, dnormal)

        # then back to cartesian at straightened position and newnormal
        new_propoints = transform_to_cartesian(radial_coords, straightened_pts[i], newNormal, newdnormal)
        
        # append transformed points
        transformed_points.append(new_propoints)
        
        if check:
            plt += mesh.alpha(0.3)
            plt += Points(propoints).ps(5).c('red')
            plt += Points(new_propoints).ps(5).c('green')

    if check:
        plt.show().close()

    # using warp function to deform the mesh based on the original and transformed points
    original_points = np.vstack(original_points)
    transformed_points = np.vstack(transformed_points)

    ## deform the mesh and show
   
    print("i am warpping now --")
    deformed_mesh = mesh.clone().warp(original_points, transformed_points, sigma=1)
    print("i am down warpping --")

    deformed_mesh.pointdata['curvature'] = curvature0 # get curvature data applied to the mesh
    vmin, vmax = getTightercmap(curvature0, sigma=1) # just to visualize the curvatures
    deformed_mesh.cmap('RdBu',deformed_mesh.pointdata['curvature'],vmin=vmin, vmax=vmax).add_scalarbar() 

    # move mesh to origin  
    Aligntransform = LinearTransform()
    Aligntransform.translate(-axispts[0])
    Aligntransform.move(deformed_mesh)


    # show([mesh,deformed_mesh],N=2,axes=1)

    deformed_mesh.write(namefile+"-straight-mesh.stl")

    dmeshpts = deformed_mesh.vertices
    dcurvs = deformed_mesh.pointdata['curvature']
    dcurvs = dcurvs.reshape(-1, 1) 
    deformedmeshvalues = np.hstack((dmeshpts, dcurvs))  

    np.save(namefile + '-deformedvalues.npy', deformedmeshvalues) # save the files
    
    return deformed_mesh

# =============================================================================
# 4. PROCESSING THE FLAT MESH
# =============================================================================

def get_flatdata(deformed_mesh, namefile=None, check=None, densitydata=None):
    """
    Flat projection from cylinderical object
    
    Args:
        mesh (mesh_object): Mesh
        axispts (array): axis points arrays
        namefile (str): Name to store the values
        check (bool): to see the plot
    
    Returns:
        radius, angle, height, demesh2values
    """

    namefile = getDefaultname(namefile)    # get name

    dmesh2 = deformed_mesh.clone().decimate(n=500).smooth(boundary=True).subdivide(n=2)
    # show([deformed_mesh, dmesh2], N=2) # check

    dmesh2pts = dmesh2.vertices
    if densitydata:
        demesh2values = dmesh2.pointdata['density']
    else:
        demesh2values = dmesh2.pointdata['curvature']

    center = np.mean(dmesh2pts, axis=0)
    normal = np.array([0, 0, 1])   # Axis of the cylinder/object
    dnormal = np.array([0, 1, 0])  # Direction of 90 degrees

    # --- Build the Rotation Matrix ---
    # Align: normal -> Z, dnormal -> Y
    z_axis = normal / np.linalg.norm(normal)
    y_axis = dnormal / np.linalg.norm(dnormal)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # R maps World -> Local
    R = np.vstack([x_axis, y_axis, z_axis])

    # --- Transform to Local Coordinates ---
    pts_centered = dmesh2pts - center
    local_pts = pts_centered @ R.T 

    # --- Convert to Polar ---
    # In our local system: 
    # local_pts[:, 0] is X, local_pts[:, 1] is Y
    radius = np.linalg.norm(local_pts[:, :2], axis=1)
    angle = np.arctan2(local_pts[:, 1], local_pts[:, 0])
    height = local_pts[:, 2]
    
    print('we got the flat mesh')
    
    combined_data = np.column_stack((radius, angle, height, dmesh2pts, demesh2values))
    np.save(namefile + '-flatdata.npy', combined_data) # save the files
    
    if check:
        visualize_flatmesh(height, angle, demesh2values)


    return radius, angle, height, dmesh2pts, demesh2values

def visualize_flatmesh(height, angle, curvature, sigma=3, namefile=None, colorstr='RdBu', densitydata=None):
    """
    Visualize Flat projection
    
    Args:
        height (array): height 
        angle (array): angle
        curvature (array): curvature
    
    Returns:
        
    """
    
    namefile = getDefaultname(namefile)    # get name
    # 1. Your data
    x = height 
    y = angle
    z = curvature

    # 2. NORMALIZE coordinates to [0, 1] range
    # This prevents the "vertical streaking" by making the axes look equal to the math
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    # 3. Create a dense grid in the normalized space [0, 1]
    grid_size = 500
    xi_norm = np.linspace(0, 1, grid_size)
    yi_norm = np.linspace(0, 1, grid_size)
    xi_norm, yi_norm = np.meshgrid(xi_norm, yi_norm)

    # 4. Interpolate using the normalized coordinates
    # 'linear' is often cleaner for large scale differences than 'cubic'
    zi = griddata((x_norm, y_norm), z, (xi_norm, yi_norm), method='linear')

    # 5. Plotting
    plt.figure(figsize=(12, 4))

    # We use the original x.min/max for the 'extent' so the axes show real units
    vmin, vmax = getTightercmap(curvature, sigma=sigma) # just to visualize the curvatures

    if densitydata:
        colorstr='Blues'
        vmin = 0
        vmax = np.max(curvature)


    # This takes your RGB array and turns it into a 'cmap' object
    cols = color_map(range(256), colorstr)
    my_cmap = mcolors.LinearSegmentedColormap.from_list("my_name", cols)

    im = plt.imshow(zi, 
                    extent=[x.min(), x.max(), y.min(), y.max()], 
                    origin='lower', 
                    aspect='auto', 
                    cmap=my_cmap,
                    vmin=vmin, vmax= vmax)

    plt.colorbar(im, label='Curvature Value')
    plt.xlabel('Height (Z-axis)')
    plt.ylabel('theta (Ventral - Dorsal - Ventral)')
    plt.title('')
        
    plt.savefig(
    namefile + "-flat.png", 
    dpi=300,                # High resolution for printing/reports
    bbox_inches='tight',    # Removes extra whitespace around the edges
    transparent=False,      # Set to True if you want a transparent background
    facecolor='white'       # Ensures background is solid white
    )
    plt.show()

def normalize_values(height, angle, curvature, shift_deg=0, namefile=None, densitydata=None):
    """Normalize all the values 

    Args:
        height (array): height 
        angle (array): angle
        curvature (array): curvature
    
    Returns:
        norm_height, angle_degrees, scaled_curvature

    """
    namefile = getDefaultname(namefile)    # get name

        
    # normalize height from 0 to 1
    norm_height = (height - np.min(height)) / (np.max(height) - np.min(height))

    # convert angle from radians to degrees
    angle_degrees = np.degrees(angle)
    angle_degrees = fix_angles(angle_degrees, shift_deg=shift_deg)

    # scale curvature
    if densitydata:
        scaled_curvature = curvature
    else:
        scaled_curvature = curvature*1e6
    
    
    # save the normalized values
    combined_normalized = np.column_stack((norm_height, angle_degrees, scaled_curvature))
    
    np.save(namefile+"-normalized-flatdata.npy", combined_normalized) # save the files
    

    return norm_height, angle_degrees, scaled_curvature

def fix_angles(angles, shift_deg=0):
    """For the times when dorsal line doesnt align with 0 degree

    Args:
        angles (array): angles in degrees
        shift_deg (int, optional): angle to displace in degrees. Defaults to 0.

    Returns:
        fixed_angles (array)
    """
    fixed_angles = angles + shift_deg
    fixed_angles = (fixed_angles + 180) % 360 - 180
    return fixed_angles



## IMPORTANT ONE IS HERE
def getDeformedmesh2(mesh, axis_info,
                    namefile=None,
                    skip_index =np.array([]),
                    dists_threshold=300,
                    check=None):
    """
    Straightening the mesh
    
    Args:
        mesh (mesh_object): Mesh
        axis_info (dictionary): axis points arrays and normals
            axis_info = {
                'axis': axispts,
                'axis normals': normals,
                'dorsal normals': dnormals
                }
        skip_index (array): number of planes which look wrong
        namefile (str): Name to store the curvature values
        dists_threshold (int): distance to exclude points when mesh is very curved
        check (bool): to see the details of process
    
    Returns:
        deformed_mesh (mesh_object): Deformed mesh
    """
    normals = axis_info['axis normals']
    dnormals = axis_info['dorsal normals']
    axispts = axis_info['axis']
    
    
    namefile = getDefaultname(namefile)    # get name
        
    mesh.c('lightblue').decimate(n=1000) # its important to decimate

    # get new normal from first two axis points
    newNormal = np.array([0, 0, 1]) #normals[0]
    newdnormal = np.array([0, 1, 0])#dnormals[0]
    # %% straighten the axis points
    # start from the first point and add distance along newNormal
    straightened_pts = [axispts[0]]
    for i in range(1, len(axispts)):
        dist = np.linalg.norm(axispts[i] - axispts[i-1])
        new_pt = straightened_pts[-1] + newNormal * dist
        straightened_pts.append(new_pt) 
    straightened_pts = np.array(straightened_pts)   

    if check:
        # convert to vedo Points for visualization
        axis = Points(axispts).c('red').ps(8)
        straightened_axis = Points(straightened_pts).c('blue').ps(8)
        plt = Plotter(axes=1)
        plt += mesh.alpha(0.3)
        plt += axis
        plt += straightened_axis

    # empty list of points in orginal mesh corresponding to each axis point and their transformed positions
    original_points = []
    transformed_points = []

    for i in range(0, len(axispts)):
    
        # if i is in skip_index, skip this iteration
        if i in skip_index:
            continue

        pt1 = axispts[i]
        normal = normals[i]
        dnormal = dnormals[i]

            
        cut = mesh.clone().intersect_with_plane(pt1, normal) 
        # cut the mesh with the plane at pt1 with normal
        propoints = cut.vertices # get the points of the cut mesh

        # if no points found, skip
        if len(propoints) == 0:
            # skip this iteration
            continue

        # check if there are two clusters of points in the propoints, by checking their distance from pt1 and then plotting a histogram
        dists = np.linalg.norm(propoints - pt1, axis=1)
        propoints = propoints[dists < dists_threshold]     # delete points that are too far away
        original_points.append(propoints)     # append original points

        # transform the propoints to radial cordinations around pt1 and normal
        radial_coords = transform_to_radial(propoints, pt1, normal, dnormal)

        # then back to cartesian at straightened position and newnormal
        new_propoints = transform_to_cartesian(radial_coords, straightened_pts[i], newNormal, newdnormal)
        
        # append transformed points
        transformed_points.append(new_propoints)
        
        if check:
            plt += mesh.alpha(0.3)
            plt += Points(propoints).ps(5).c('red')
            plt += Points(new_propoints).ps(5).c('green')

    if check:
        plt.show().close()

    # using warp function to deform the mesh based on the original and transformed points
    original_points = np.vstack(original_points)
    transformed_points = np.vstack(transformed_points)

    ## deform the mesh and show
    print(mesh)
    print("i am warpping now --")
    deformed_mesh = mesh.clone().warp(original_points, transformed_points, sigma=1)
    print("i am down warpping --")
    print(deformed_mesh)

    # move mesh to origin  
    Aligntransform = LinearTransform()
    Aligntransform.translate(-axispts[0])
    Aligntransform.move(deformed_mesh)


    # show([mesh,deformed_mesh],N=2,axes=1)
    
    return deformed_mesh



def get_flatdata2(deformed_mesh, namefile=None, check=None):
    """
    Flat projection from cylinderical object
    
    Args:
        mesh (mesh_object): Mesh
        axispts (array): axis points arrays
        namefile (str): Name to store the values
        check (bool): to see the plot
    
    Returns:
        radius, angle, height, demesh2values
    """

    namefile = getDefaultname(namefile)    # get name

    dmesh2 = deformed_mesh.clone().decimate(n=500).smooth(boundary=True).subdivide(n=2)
    # show([deformed_mesh, dmesh2], N=2) # check

    dmesh2pts = dmesh2.vertices
    
    center = np.mean(dmesh2pts, axis=0)
    normal = np.array([0, 0, 1])   # Axis of the cylinder/object
    dnormal = np.array([0, 1, 0])  # Direction of 90 degrees

    # --- Build the Rotation Matrix ---
    # Align: normal -> Z, dnormal -> Y
    z_axis = normal / np.linalg.norm(normal)
    y_axis = dnormal / np.linalg.norm(dnormal)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # R maps World -> Local
    R = np.vstack([x_axis, y_axis, z_axis])

    # --- Transform to Local Coordinates ---
    pts_centered = dmesh2pts - center
    local_pts = pts_centered @ R.T 

    # --- Convert to Polar ---
    # In our local system: 
    # local_pts[:, 0] is X, local_pts[:, 1] is Y
    radius = np.linalg.norm(local_pts[:, :2], axis=1)
    angle = np.arctan2(local_pts[:, 1], local_pts[:, 0])
    height = local_pts[:, 2]
    
    print('we got the flat mesh')
    

    return radius, angle, height, dmesh2







def visualize_flatmesh2(height, angle, data,
    sigma=3, namefile=None,
    colorstr='RdBu'):
    """
    Visualize Flat projection
    
    Args:
        height (array): height 
        angle (array): angle
        curvature (array): curvature
    
    Returns:
        
    """
    
    namefile = getDefaultname(namefile)    # get name
    # 1. Your data
    x = height 
    y = angle
    z = data

    # 2. NORMALIZE coordinates to [0, 1] range
    # This prevents the "vertical streaking" by making the axes look equal to the math
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    # 3. Create a dense grid in the normalized space [0, 1]
    grid_size = 500
    xi_norm = np.linspace(0, 1, grid_size)
    yi_norm = np.linspace(0, 1, grid_size)
    xi_norm, yi_norm = np.meshgrid(xi_norm, yi_norm)

    # 4. Interpolate using the normalized coordinates
    # 'linear' is often cleaner for large scale differences than 'cubic'
    zi = griddata((x_norm, y_norm), z, (xi_norm, yi_norm), method='linear')

    # 5. Plotting
    plt.figure(figsize=(12, 4))

    # We use the original x.min/max for the 'extent' so the axes show real units
    vmin, vmax = getTightercmap(data, sigma=sigma) # just to visualize the curvatures


    # This takes your RGB array and turns it into a 'cmap' object
    cols = color_map(range(256), colorstr)
    my_cmap = mcolors.LinearSegmentedColormap.from_list("my_name", cols)

    im = plt.imshow(zi, 
                    extent=[x.min(), x.max(), y.min(), y.max()], 
                    origin='lower', 
                    aspect='auto', 
                    cmap=my_cmap,
                    vmin=vmin, vmax= vmax)

    plt.colorbar(im, label='Value')
    plt.xlabel('Normalized Rostral-Caudal Distance')
    plt.ylabel('Ventral - Dorsal - Ventral')
    plt.title('')
        
    # plt.savefig(
    # namefile + "-flat.png", 
    # dpi=300,                # High resolution for printing/reports
    # bbox_inches='tight',    # Removes extra whitespace around the edges
    # transparent=False,      # Set to True if you want a transparent background
    # facecolor='white'       # Ensures background is solid white
    # )
    plt.show()


def normalize_values2(height, angle, shift_deg=0):
    """Normalize all the values 

    Args:
        height (array): height 
        angle (array): angle
        curvature (array): curvature
    
    Returns:
        norm_height, angle_degrees, scaled_curvature

    """
        
    # normalize height from 0 to 1
    norm_height = (height - np.min(height)) / (np.max(height) - np.min(height))

    # convert angle from radians to degrees
    angle_degrees = np.degrees(angle)
    angle_degrees = fix_angles(angle_degrees, shift_deg=shift_deg)

    return norm_height, angle_degrees



    