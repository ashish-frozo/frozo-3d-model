"""
Point Cloud to Mesh Conversion

Converts completed point clouds to meshes for AR/web viewing.
Supports:
- Ball Pivoting Algorithm (BPA) - primary
- Poisson Surface Reconstruction - fallback

Usage:
    from svscn.inference import point_cloud_to_mesh
    mesh = point_cloud_to_mesh(points)
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MeshResult:
    """Result from mesh reconstruction."""
    
    vertices: np.ndarray   # (V, 3) vertex positions
    faces: np.ndarray      # (F, 3) face indices
    normals: Optional[np.ndarray] = None  # (V, 3) vertex normals
    
    # Quality metrics
    num_vertices: int = 0
    num_faces: int = 0
    is_watertight: bool = False
    surface_area: float = 0.0
    
    # Method used
    method: str = "unknown"


def point_cloud_to_mesh(
    points: np.ndarray,
    method: str = "ball_pivoting",
    simplify: bool = True,
    target_faces: int = 50000
) -> Optional[MeshResult]:
    """
    Convert point cloud to mesh.
    
    Args:
        points: (N, 3) point cloud
        method: "ball_pivoting" or "poisson"
        simplify: Whether to simplify high-poly meshes
        target_faces: Target face count for simplification
    
    Returns:
        MeshResult or None if failed
    """
    try:
        import open3d as o3d
    except ImportError:
        logger.error("open3d not installed. Run: pip install open3d")
        return None
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estimate normals (required for meshing)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    # Orient normals consistently
    pcd.orient_normals_consistent_tangent_plane(k=10)
    
    # Reconstruct mesh
    if method == "ball_pivoting":
        mesh = _ball_pivoting_reconstruction(pcd)
    elif method == "poisson":
        mesh = _poisson_reconstruction(pcd)
    else:
        logger.error(f"Unknown method: {method}")
        return None
    
    if mesh is None:
        return None
    
    # Post-process
    mesh = _clean_mesh(mesh)
    
    # Simplify if needed
    if simplify and len(mesh.triangles) > target_faces:
        mesh = mesh.simplify_quadric_decimation(target_faces)
    
    # Compute normals
    mesh.compute_vertex_normals()
    
    # Extract arrays
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.vertex_normals)
    
    return MeshResult(
        vertices=vertices,
        faces=faces,
        normals=normals,
        num_vertices=len(vertices),
        num_faces=len(faces),
        is_watertight=mesh.is_watertight(),
        surface_area=mesh.get_surface_area(),
        method=method
    )


def _ball_pivoting_reconstruction(pcd) -> Optional["o3d.geometry.TriangleMesh"]:
    """
    Ball Pivoting Algorithm reconstruction.
    
    Good for point clouds with uniform density.
    """
    import open3d as o3d
    
    try:
        # Compute average distance for radius estimation
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        
        # Ball pivoting radii (multiple scales)
        radii = [avg_dist * r for r in [0.5, 1.0, 2.0, 4.0]]
        
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector(radii)
        )
        
        if len(mesh.triangles) < 10:
            logger.warning("BPA produced too few faces, trying Poisson")
            return _poisson_reconstruction(pcd)
        
        return mesh
        
    except Exception as e:
        logger.warning(f"BPA failed: {e}, falling back to Poisson")
        return _poisson_reconstruction(pcd)


def _poisson_reconstruction(pcd, depth: int = 9) -> Optional["o3d.geometry.TriangleMesh"]:
    """
    Poisson Surface Reconstruction.
    
    Produces smoother surfaces, good for noisy data.
    """
    import open3d as o3d
    
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=depth,
            linear_fit=True
        )
        
        # Remove low-density vertices (outside point cloud)
        densities = np.asarray(densities)
        density_threshold = np.percentile(densities, 5)
        
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        return mesh
        
    except Exception as e:
        logger.error(f"Poisson reconstruction failed: {e}")
        return None


def _clean_mesh(mesh) -> "o3d.geometry.TriangleMesh":
    """Clean up mesh artifacts."""
    import open3d as o3d
    
    # Remove degenerate triangles
    mesh.remove_degenerate_triangles()
    
    # Remove duplicate triangles
    mesh.remove_duplicated_triangles()
    
    # Remove duplicate vertices
    mesh.remove_duplicated_vertices()
    
    # Remove non-manifold edges
    mesh.remove_non_manifold_edges()
    
    # Remove unreferenced vertices
    mesh.remove_unreferenced_vertices()
    
    return mesh


def repair_mesh(mesh_result: MeshResult) -> MeshResult:
    """
    Attempt to repair a mesh with holes or artifacts.
    
    Uses hole filling and smoothing.
    """
    try:
        import trimesh
        
        mesh = trimesh.Trimesh(
            vertices=mesh_result.vertices,
            faces=mesh_result.faces
        )
        
        # Fill small holes
        trimesh.repair.fill_holes(mesh)
        
        # Fix normals
        mesh.fix_normals()
        
        # Remove degenerate faces
        mesh.remove_degenerate_faces()
        
        return MeshResult(
            vertices=mesh.vertices,
            faces=mesh.faces,
            normals=mesh.vertex_normals,
            num_vertices=len(mesh.vertices),
            num_faces=len(mesh.faces),
            is_watertight=mesh.is_watertight,
            surface_area=mesh.area,
            method=mesh_result.method + "_repaired"
        )
        
    except Exception as e:
        logger.warning(f"Mesh repair failed: {e}")
        return mesh_result


def add_texture_from_image(
    mesh_result: MeshResult,
    image_path: Path,
    uv_method: str = "projection"
) -> Tuple[MeshResult, np.ndarray]:
    """
    Add texture to mesh from original image.
    
    Args:
        mesh_result: Mesh to texture
        image_path: Path to original image
        uv_method: UV mapping method
    
    Returns:
        (mesh, uv_coords) tuple
    
    Note: Full texture projection requires depth estimation.
    This is a placeholder for v2.
    """
    # Placeholder for v2 implementation
    logger.info("Texture projection is a v2 feature")
    
    # Simple planar projection for now
    vertices = mesh_result.vertices
    
    # Project to XY plane for UVs
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    
    # Normalize to [0, 1]
    u = (x - x.min()) / (x.max() - x.min() + 1e-6)
    v = (y - y.min()) / (y.max() - y.min() + 1e-6)
    
    uv_coords = np.stack([u, v], axis=1)
    
    return mesh_result, uv_coords


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert point cloud to mesh")
    parser.add_argument("--input", type=Path, required=True, help="Input .npy file")
    parser.add_argument("--output", type=Path, required=True, help="Output mesh file")
    parser.add_argument("--method", type=str, default="ball_pivoting")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Load points
    points = np.load(args.input)
    
    # Convert
    result = point_cloud_to_mesh(points, method=args.method)
    
    if result:
        print(f"Vertices: {result.num_vertices}")
        print(f"Faces: {result.num_faces}")
        print(f"Watertight: {result.is_watertight}")
        
        # Save
        import trimesh
        mesh = trimesh.Trimesh(vertices=result.vertices, faces=result.faces)
        mesh.export(args.output)
        print(f"Saved to: {args.output}")
    else:
        print("Mesh reconstruction failed")
