#!/bin/bash

xsetroot -solid "#333333"
(sleep 2s && tint2 -c /etc/xdg/tint2/tint2rc) &
openbox
