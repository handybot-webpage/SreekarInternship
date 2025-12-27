Live_infer.py should be ran once Godot exe file is installed
The model folder is empty, a model folder called models should be created and the pre trained model should be kept in there
Cam index will check which camera should be used to send the frames to the model. In this case, the default index is set as 0 but it can be added by one time if an another camera is added. 

## Pip installations
Windows host Powershell: 
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python
pip install pillow matplotlib numpy pandas

Pytorch CUDA Docker Container:
python -m pip install --upgrade pip
pip install opencv-python-headless
pip install pillow matplotlib numpy pandas