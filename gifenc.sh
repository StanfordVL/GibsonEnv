palette="/tmp/palette$1.png"
reverse="/tmp/reverse$1.avi"
filters="fps=50,scale=-1:-1:flags=lanczos"

ffmpeg -v warning -i $1 -vf "$filters,palettegen" -y $palette
ffmpeg -v warning -i $1 -i $palette -lavfi "$filters [x]; [x][1:v] paletteuse" -y $2


ffmpeg -i $1 -vf reverse $reverse
ffmpeg -v warning -i $reverse -vf "$filters,palettegen" -y $palette
ffmpeg -v warning -i $reverse -i $palette -lavfi "$filters [x]; [x][1:v] paletteuse" -y $3

