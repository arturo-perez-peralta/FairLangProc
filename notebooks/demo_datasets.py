import os, sys, json
from pathlib import Path

# Asumiendo que tu paquete 'FairLangProc' está en el directorio raíz
# y que este script está en una carpeta 'demos'
try:
    from FairLangProc.datasets import BiasDataLoader
except ImportError:
    print("Error: No se pudo importar 'FairLangProc'. Asegúrate de que el paquete esté instalado o la ruta sea correcta.")
    sys.exit(1)


def setup_path(config):
    """Configura el sys.path si se ejecuta en modo LOCAL."""
    if config.get("LOCAL", False):
        root_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), config.get("ROOT_PATH_RELATIVE", ".."))
        )
        if root_path not in sys.path:
            sys.path.insert(0, root_path)
            print(f"Añadido al sys.path: {root_path}")
    # Esta importación debe ocurrir DESPUÉS de configurar la ruta
    from FairLangProc.datasets import BiasDataLoader
    return BiasDataLoader

def load_config(config_path="config_demos.json"):
    """Carga el archivo de configuración JSON."""
    with open(config_path, "r") as f:
        return json.load(f)

def demo_bbq(loader):
    """Demostración del dataset BBQ."""
    print("\n--- Demostración: BBQ (raw) ---")
    age_bbq_raw = loader(dataset='BBQ', config='Age', format='raw')
    print(age_bbq_raw['data'].head())
    
    print("\n--- Demostración: BBQ (default) ---")
    age_bbq = loader(dataset='BBQ', config='Age', format='default')
    print(age_bbq['data'])

def demo_stereoset(loader):
    """Demostración del dataset StereoSet."""
    print("\n--- Demostración: StereoSet (raw) ---")
    stereo_raw = loader(dataset='StereoSet', config='Gender', format='raw')
    print(stereo_raw['data'].head())
    
    print("\n--- Demostración: StereoSet (default) ---")
    stereo = loader(dataset='StereoSet', config='Gender', format='default')
    print(stereo['data'])

def demo_crows_pairs(loader):
    """Demostración del dataset CrowS-Pairs."""
    print("\n--- Demostración: CrowS-Pairs (raw) ---")
    crows_raw = loader(dataset='CrowsPairs', format='raw')
    print(crows_raw['data'].head())

def demo_winogender(loader):
    """Demostración del dataset WinoGender."""
    print("\n--- Demostración: WinoGender (raw) ---")
    wino_raw = loader(dataset='WinoGender', format='raw')
    print(wino_raw['data'].head())

def main():
    """Ejecuta todas las demostraciones de datasets."""
    config = load_config()
    
    # Configura la ruta e importa el loader
    try:
        BiasDataLoader = setup_path(config)
    except ImportError as e:
        print(f"Error de importación: {e}")
        print("Asegúrate de que 'FairLangProc' esté en el directorio superior o instalado.")
        return

    demo_bbq(BiasDataLoader)
    demo_stereoset(BiasDataLoader)
    demo_crows_pairs(BiasDataLoader)
    demo_winogender(BiasDataLoader)

if __name__ == "__main__":
    main()