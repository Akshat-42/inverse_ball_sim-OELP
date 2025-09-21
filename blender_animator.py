import bpy # type: ignore
import json
import os

def setup_and_animate(ball_name="CricketBall", filename="trajectory.json"):
    
    if ball_name not in bpy.data.objects:
        print(f"Object '{ball_name}' not found. Creating a new one.")
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.035, 
            location=(0, 0, 0)
        )
        # The new sphere is automatically the active object
        ball = bpy.context.active_object
        ball.name = ball_name
        # Make it look smooth instead of faceted
        bpy.ops.object.shade_smooth()
    else:
        print(f"Found existing object named '{ball_name}'.")
        ball = bpy.data.objects[ball_name]

    # --- Load the trajectory data ---
    blend_dir = os.path.dirname(bpy.data.filepath)
    if not blend_dir:
        print("Error: Please save your .blend file first.")
        # The os.path.dirname command needs a saved file to know the directory.
        return

    filepath = os.path.join(blend_dir, filename)
    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'.")
        print("Ensure 'trajectory.json' is in the same folder as your .blend file.")
        return

    with open(filepath, "r") as f:
        trajectory_data = json.load(f)

    # --- Clear existing animation ---
    ball.animation_data_clear()

    sps = 60
    fps = 60

    # --- Create keyframes ---
    for frame_data in trajectory_data:
        frame_num = frame_data["sample"]//(sps//fps)
        pos = frame_data["position"]
        

        ball.location = pos

        ball.keyframe_insert(data_path="location", frame=frame_num)
        
    if trajectory_data:
        bpy.context.scene.frame_end = trajectory_data[-1]["sample"]//(sps//fps)
        
    print(f"Animation complete for '{ball_name}' with {len(trajectory_data)} keyframes.")

# --- Run the main function ---
setup_and_animate()