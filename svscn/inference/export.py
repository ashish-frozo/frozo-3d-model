"""
3D Asset Export Utilities

Export completed point clouds and meshes to standard formats:
- OBJ: Universal, for editing
- GLB: Web/AR optimized
- PLY: Point cloud format

Usage:
    from svscn.inference import export_mesh
    export_mesh(mesh_result, "output.glb")
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .mesh_utils import MeshResult

logger = logging.getLogger(__name__)


def export_mesh(
    mesh: MeshResult,
    output_path: Union[str, Path],
    format: Optional[str] = None
) -> bool:
    """
    Export mesh to file.
    
    Args:
        mesh: MeshResult from mesh reconstruction
        output_path: Output file path
        format: Force format, or auto-detect from extension
    
    Returns:
        True if successful
    """
    output_path = Path(output_path)
    
    if format is None:
        format = output_path.suffix.lower().lstrip(".")
    
    try:
        import trimesh
        
        # Create trimesh object
        tm = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            vertex_normals=mesh.normals
        )
        
        # Export based on format
        if format in ["glb", "gltf"]:
            tm.export(output_path, file_type="glb")
        elif format == "obj":
            tm.export(output_path, file_type="obj")
        elif format == "ply":
            tm.export(output_path, file_type="ply")
        elif format == "stl":
            tm.export(output_path, file_type="stl")
        else:
            logger.warning(f"Unknown format {format}, using OBJ")
            tm.export(output_path.with_suffix(".obj"), file_type="obj")
        
        logger.info(f"Exported mesh to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return False


def export_pointcloud(
    points: np.ndarray,
    output_path: Union[str, Path],
    colors: Optional[np.ndarray] = None,
    format: Optional[str] = None
) -> bool:
    """
    Export point cloud to file.
    
    Args:
        points: (N, 3) point positions
        output_path: Output file path
        colors: Optional (N, 3) RGB colors [0-255]
        format: ply, pcd, or xyz
    
    Returns:
        True if successful
    """
    output_path = Path(output_path)
    
    if format is None:
        format = output_path.suffix.lower().lstrip(".")
    
    try:
        import open3d as o3d
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
            # Normalize to [0, 1] if needed
            if colors.max() > 1:
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Estimate normals for better visualization
        pcd.estimate_normals()
        
        if format == "ply":
            o3d.io.write_point_cloud(str(output_path), pcd)
        elif format == "pcd":
            o3d.io.write_point_cloud(str(output_path), pcd)
        elif format == "xyz":
            # Simple text format
            np.savetxt(output_path, points, fmt="%.6f")
        elif format == "npy":
            np.save(output_path, points)
        else:
            logger.warning(f"Unknown format {format}, using PLY")
            o3d.io.write_point_cloud(str(output_path.with_suffix(".ply")), pcd)
        
        logger.info(f"Exported point cloud to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Point cloud export failed: {e}")
        return False


def export_for_ar(
    mesh: MeshResult,
    output_dir: Union[str, Path],
    name: str = "model"
) -> dict:
    """
    Export mesh in formats optimized for AR viewers.
    
    Creates:
    - GLB for web/Android
    - USDZ for iOS (if available)
    
    Args:
        mesh: MeshResult
        output_dir: Output directory
        name: Model name
    
    Returns:
        Dict of format -> path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exports = {}
    
    # GLB (universal)
    glb_path = output_dir / f"{name}.glb"
    if export_mesh(mesh, glb_path):
        exports["glb"] = glb_path
    
    # OBJ (fallback)
    obj_path = output_dir / f"{name}.obj"
    if export_mesh(mesh, obj_path):
        exports["obj"] = obj_path
    
    # Generate USDZ for iOS (requires usd-core)
    try:
        usdz_path = _create_usdz(mesh, output_dir / f"{name}.usdz")
        if usdz_path:
            exports["usdz"] = usdz_path
    except Exception as e:
        logger.warning(f"USDZ creation not available: {e}")
    
    return exports


def _create_usdz(mesh: MeshResult, output_path: Path) -> Optional[Path]:
    """
    Create USDZ file for iOS AR.
    
    Requires usd-core or similar USD library.
    """
    # Check for USD availability
    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        logger.info("USD libraries not available. Install usd-core for USDZ support.")
        return None
    
    try:
        # Create USD stage
        stage = Usd.Stage.CreateNew(str(output_path.with_suffix(".usda")))
        
        # Create mesh
        mesh_path = "/Model/Mesh"
        usd_mesh = UsdGeom.Mesh.Define(stage, mesh_path)
        
        # Set vertices
        usd_mesh.CreatePointsAttr(mesh.vertices.tolist())
        
        # Set faces (USD needs flat list with face vertex counts)
        face_counts = [3] * len(mesh.faces)
        face_indices = mesh.faces.flatten().tolist()
        
        usd_mesh.CreateFaceVertexCountsAttr(face_counts)
        usd_mesh.CreateFaceVertexIndicesAttr(face_indices)
        
        # Set normals
        if mesh.normals is not None:
            usd_mesh.CreateNormalsAttr(mesh.normals.tolist())
        
        stage.Save()
        
        # Convert to USDZ
        # This requires usdzconvert tool or similar
        logger.info(f"USD file created: {output_path.with_suffix('.usda')}")
        logger.info("Convert to USDZ using: usdzconvert model.usda model.usdz")
        
        return output_path.with_suffix(".usda")
        
    except Exception as e:
        logger.error(f"USDZ creation failed: {e}")
        return None


def create_preview_thumbnail(
    mesh: MeshResult,
    output_path: Union[str, Path],
    size: tuple = (512, 512),
    background: str = "white"
) -> bool:
    """
    Render thumbnail image of mesh.
    
    Useful for preview before download.
    """
    try:
        import open3d as o3d
        
        # Create mesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d_mesh.compute_vertex_normals()
        
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=size[0], height=size[1], visible=False)
        vis.add_geometry(o3d_mesh)
        
        # Set view
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        
        # Render
        vis.poll_events()
        vis.update_renderer()
        
        # Capture
        image = vis.capture_screen_float_buffer(do_render=True)
        
        vis.destroy_window()
        
        # Save
        import numpy as np
        from PIL import Image
        
        img_array = (np.asarray(image) * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        img.save(output_path)
        
        logger.info(f"Thumbnail saved to {output_path}")
        return True
        
    except Exception as e:
        logger.warning(f"Thumbnail creation failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export 3D assets")
    parser.add_argument("--input", type=Path, required=True, help="Input mesh or point cloud")
    parser.add_argument("--output", type=Path, required=True, help="Output path")
    parser.add_argument("--format", type=str, default=None)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Determine input type
    suffix = args.input.suffix.lower()
    
    if suffix == ".npy":
        # Point cloud
        points = np.load(args.input)
        export_pointcloud(points, args.output, format=args.format)
    else:
        # Assume mesh
        import trimesh
        tm = trimesh.load(args.input)
        mesh = MeshResult(
            vertices=np.array(tm.vertices),
            faces=np.array(tm.faces),
            num_vertices=len(tm.vertices),
            num_faces=len(tm.faces)
        )
        export_mesh(mesh, args.output, format=args.format)
