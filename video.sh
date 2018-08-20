yes | ffmpeg -framerate 45 -i output/F%07d.ppm -vcodec libx264 -b:v 4000k video/animation.mp4
