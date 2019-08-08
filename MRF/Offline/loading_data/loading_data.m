
fid = textread('urls.txt', '%s', 'delimiter', ',');


for k=1:400
	for i=1:2^(16)
	    filename = 'data'+string(i)+'.mat';
	    outfilename = websave(filename,url,weboptions('ContentType','binary'));
	    x = load(filename);
	    save(x);
	    disp(i);
	end