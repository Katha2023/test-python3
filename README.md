# README

* Clone the repository:
```bash
git clone https://github.com/Katha2023/test-python3
cd test-python3
```
* Run the script:
```bash
chmod u+x activate_env.sh
./activate_env.sh
```
* Save the png file to remote in local machine from the directory containing ``password.pem``:
```pwsh
scp -i "password.pem" ubuntu@ec2-X-Y-Z-A.ap-south-1.compute.amazonaws.com:/home/ubuntu/test-python3/plot.png plot.png
```
where ``X.Y.Z.A`` is the instance public IP address. Make sure to check the name of the png file before copying.
* Modify ``activate_env.sh`` to decide which Python file to run:
```bash
python3 <your_file_name>
```
* If you make any changes to any file via GitHub, make sure to ``git pull`` while inside the ``test-python3`` directory.
