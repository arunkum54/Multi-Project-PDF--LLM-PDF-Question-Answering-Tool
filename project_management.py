import os
import shutil

BASE_DIR = "projects"

def create_project(project_name):
    """
    Create a new project directory.
    """
    project_dir = os.path.join(BASE_DIR, project_name)
    os.makedirs(project_dir, exist_ok=True)
    print(f"Project '{project_name}' created successfully.")

def list_projects():
    """
    List all existing projects.
    """
    if not os.path.exists(BASE_DIR):
        print("No projects exist.")
        return []
    return [name for name in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, name))]

def delete_project(project_name):
    """
    Delete a project directory.
    """
    project_dir = os.path.join(BASE_DIR, project_name)
    if os.path.exists(project_dir):
        shutil.rmtree(project_dir)
        print(f"Project '{project_name}' deleted successfully.")
    else:
        print(f"Project '{project_name}' does not exist.")
