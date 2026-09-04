import shutil
import sys

def restore_model(backup_filename):
    """
    Restores a specific backed-up model as the live 'base_model.pkl'.
    Usage: python src/restore_model.py base_model_20260830_235800.pkl
    """
    backup_path = f'models/history/{backup_filename}'
    shutil.copy(backup_path, 'models/base_model.pkl')
    print(f"Restored {backup_path} -> models/base_model.pkl")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python src/restore_model.py <backup_filename>")
        sys.exit(1)
    restore_model(sys.argv[1])