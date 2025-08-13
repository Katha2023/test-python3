sudo apt update
sudo apt install python3-pip python3-venv

python3 -m venv myenv
source myenv/bin/activate
pip3 install -r requirements

taus=$(python3 -c "import numpy as np; print(' '.join(str(t) for t in np.array([350, 400])))")
#taus=$(python3 -c "import numpy as np; print(' '.join(str(t) for t in np.linspace(300, 400, 25)))")
for tau in $taus; do
    echo "Running tau=$tau..."
    python3 square_wave_single_tau.py "$tau"
done
