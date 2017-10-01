tic;
img = imread('~/Downloads/save.bmp');
s = sum(img, 3);
nz  = s>0;
[x,y] = meshgrid(1:2048, 1:1024);
xnz = x(nz);
ynz = y(nz);
r = img(:,:,1);
g = img(:,:,2);
b = img(:,:,3);
rnz = double(r(nz));
gnz = double(g(nz));
bnz = double(b(nz));
[xq,yq] = meshgrid(1:2048, 1:1024);
rq = griddata(xnz,ynz,rnz,xq,yq);
gq = griddata(xnz,ynz,gnz,xq,yq);
bq = griddata(xnz,ynz,bnz,xq,yq);

rq = reshape(rq, [1024, 2048, 1]);
gq = reshape(gq, [1024, 2048, 1]);
bq = reshape(bq, [1024, 2048, 1]);

imgq = uint8(cat(3, rq, gq, bq));

toc;