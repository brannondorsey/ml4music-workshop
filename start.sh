# 1. Creates container and attaches if none exists. or...
# 2. If container already exists, but is stopped, start it and attach. or...
# 3. If container already exists and is running, attach.

# usage: ./start.sh [--gpu]
# 	if --gpu flag is provided, nvidia-docker will be used instead of docker

IMAGE_NAME="brannondorsey/ml4music-workshop"
DOCKER="docker" 

if [ "$1" ] && [ "$1" == "--gpu" ]; then
	echo "Using nvidia-docker..."
	DOCKER="nvidia-docker"
fi

if [ ! "$(docker ps -a | grep $IMAGE_NAME)" ]; then
	echo "Container does not exist, creating one..."
	$DOCKER run -it -p 7006:7006 -p 7007:7007 -p 7008:7008 $IMAGE_NAME
else
	if [ ! "$(docker ps | grep $IMAGE_NAME)" ]; then
			echo "Container exists and is not running, starting container..."
			CONTAINER_ID=$(docker ps -aqf "ancestor=$IMAGE_NAME")
			$DOCKER start $CONTAINER_ID
	fi
	echo "Attaching to container..."
	CONTAINER_ID=$(docker ps -qf "ancestor=$IMAGE_NAME")
	$DOCKER exec -it $CONTAINER_ID bash
fi
