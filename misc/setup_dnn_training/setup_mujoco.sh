echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/.mujoco/mujoco210/bin' >> ~/.bashrc 
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia' >> ~/.bashrc 

sudo apt-get update -y 
sudo apt-get install -y libglew-dev patchelf

source ~/.bashrc