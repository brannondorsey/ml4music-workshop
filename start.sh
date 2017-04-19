# 1. Creates container and attaches if none exists. or...
# 2. If container already exists, but is stopped, start it and attach. or...
# 3. If container already exists and is running, attach.

# usage: ./start.sh [--gpu] [--root]
# 	if --gpu flag is provided, nvidia-docker will be used instead of docker
#   if --root flag is provided you will be logged into the container as root

IMAGE_NAME="brannondorsey/ml4music-workshop"
DOCKER="docker"
USER="docker"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if ( [ "$1" ] && [ "$1" == "--gpu" ] ) || ( [ "$2" ] && [ "$2" == "--gpu" ] )
then
	echo "Using nvidia-docker..."
	DOCKER="nvidia-docker"
fi

if ( [ "$1" ] && [ "$1" == "--root" ] ) || ( [ "$2" ] && [ "$2" == "--root" ] )
then
	echo "Logging in as root..."
	USER="root"
fi

if [ ! "$(docker ps -a | grep $IMAGE_NAME)" ]
then
	echo "Container does not exist, creating one..."
	$DOCKER run -it -p 7006:7006 -p 7007:7007 -p 7008:7008 --user $USER \
		-v $DIR/data:/home/docker/ml4music-workshop/data $IMAGE_NAME
else
	if [ ! "$(docker ps | grep $IMAGE_NAME)" ]
	then
			echo "Container exists and is not running, starting container..."
			CONTAINER_ID=$(docker ps -aqf "ancestor=$IMAGE_NAME")
			$DOCKER start $CONTAINER_ID
	fi
	echo "Attaching to container..."
	CONTAINER_ID=$(docker ps -qf "ancestor=$IMAGE_NAME")
	$DOCKER exec -it --user $USER $CONTAINER_ID bash
fi
