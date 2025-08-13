sudo apt update
sudo apt install python3-pip python3-venv

python3 -m venv myenv
source myenv/bin/activate

pip3 install -r requirements
python3 square_wave_opt.py
