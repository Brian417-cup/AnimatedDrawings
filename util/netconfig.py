remote_server_ip = "localhost"
remote_server_port = "8000"
base_url = f"http://{remote_server_ip}:{remote_server_port}"

static_res_url = f"{base_url}/static"
static_drawing_res_url = f"{static_res_url}/drawing_demo"
static_joint_res_url = f"{static_res_url}/joint_demo"
static_triangle_res_url = f"{static_res_url}/triangle_demo"
static_bvh_res_url = f"{static_res_url}/motion_demo"

upload_sketch_url = f"{base_url}/upload"
acquire_animation_url = f"{base_url}/animation"
