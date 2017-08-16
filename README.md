# Real environment for semantic planning project 
## Note
This is a 0.0.1 alpha release, for use in Stanford SVL only. 

## Demo

Here is a demo of a human controlled agent navigating through a virtual environment. 
![demo](https://github.com/fxia22/realenv/blob/full_environment2/misc/example.gif)

## Setup 


### Server side
- Server side uses XVnc4 as vnc server. In order to use, first `git clone` this repository and go into root directory, then create a password first with `vncpasswd pw`.
- You will also need a model file to render the views, contact feixia@stanford.edu to obtain the model. Replace the path in `init.sh` with path to the model.
- Build renderer with `./build.sh`
- Run `init.sh`, this will run the rendering engine and vncserver.
- Connect with the client to 5901 port. This can also be configured in `init.sh`.

As a demo, a server is running at capri19.stanford.edu:5901, contact feixia@stanford.edu to obtain the password. 


### Client side
TBA
