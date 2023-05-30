import argparse, os
from tensorflow import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef

model_path = "model"
logs_path = "logs/inspect"

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model_name', type=str, help='Model name', required=True)
    args = ap.parse_args()

    if not os.path.exists(os.path.join(model_path, args.model_name+".h5")):
        print("Model name not valid")

    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    with CustomObjectScope({'dice_loss': dice_loss, 'dice_coef': dice_coef}):
            model = load_model(os.path.join(model_path, args.model_name+".h5"))
    from tensorflow_model_optimization.quantization.keras import vitis_inspect
    # Set Ultra96 fingerprint as target
    inspector = vitis_inspect.VitisInspector(target="0x101000016010404")
    inspector.inspect_model(
        model, 
        plot=True, 
        plot_file=os.path.join(logs_path, args.model_name+".svg"), 
        dump_results=True, 
        dump_results_file=os.path.join(logs_path, args.model_name+".txt"), 
        verbose=1
    )
