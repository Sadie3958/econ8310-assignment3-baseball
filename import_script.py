import torch
from assignment_script import get_baseball_model

def evaluate_model(weight_path="baseball_weights.pth"):
    # Initialize the architecture built in script py
    model = get_baseball_model()
    
    # Load the learned weights (the 44MB file) (which is in the released section if you can't find it)
    try:
        model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
        model.eval()
        print("Model weights loaded successfully. Ready for evaluation.")
        return model
    except FileNotFoundError:
        print("Error: 'baseball_weights.pth' not found. Ensure it is in the same directory.")

if __name__ == "__main__":
    evaluate_model()
