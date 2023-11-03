docker build -t digits:v1 -f docker/Dockerfile .
docker volume create mlvolume
docker run -d --name mlcontainer -v mlvolume:/digits/models digits:v1
docker cp mlcontainer:/digits/models/ \\wsl.localhost\Ubuntu\home\ankit07\ScikitDigits\models