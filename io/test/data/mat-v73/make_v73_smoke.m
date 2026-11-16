function make_v73_smoke(d, ncols)
% Synthetic Brainstorm-like variables saved twice by MATLAB itself: t73.mat (-v7.3) and t7.mat (-v7), plus utf*.mat and opaque73.mat.
% Used by cpp/test (nxr-io-bst-compare reads the pair and requires them equal). Run: matlab -batch "make_v73_smoke(pwd, 200)"
if nargin < 2, ncols = 200; end
rng(1);
Vertices = rand(500,3); Faces = int32(randi(500,[900 3])); Comment=['pial ' char(233) ' ' char(252) ' ' char(55357) char(56832)]; 
sc1 = struct('Vertices',uint32([1 2 3]),'Label','A','Color',[1 0 0],'Region','LF','Function','Mean','Handles',[]);
sc2 = struct('Vertices',uint32([4 5]),'Label','B','Color',[0 1 0],'Region','RT','Function','Mean','Handles',[]);
at(1).Name='Desikan'; at(1).Scouts=[sc1 sc2];
at(2).Name='Empty'; at(2).Scouts=repmat(sc1,0,1);
at(3).Name='Cell'; at(3).Scouts=sc1;
Atlas = at;
Reg.Sphere.Vertices = rand(500,3); Reg.Sphere.Comment={'a','bb'};
tess = struct('Vertices',Vertices,'Faces',Faces,'Comment',Comment,'Atlas',Atlas,'Reg',Reg,'iAtlas',2,'Nothing',[],'Str','');
big = single(repmat(linspace(0,1,ncols),300,1) + repmat((1:300)'/1000,1,ncols));
sp = sparse([1 5 9 3],[2 2 7 9],[1.5 2.5 -3 4],10,12);
spc = sparse([1 2],[1 3],[1+2i 3-4i],4,4);
cs = {'one','two',[],{'nested',3}; 1,2,uint8([1 2 3]),'x'};
lg = logical([1 0 1; 0 0 1]); lgs = true;
cx = [1+2i 3-4i; 5 6i]; cxs = single([1+1i 2]);
ec = ''; ec0 = zeros(0,3); emptycell = {}; emptystruct = struct([]);
chm = ['abc';'def']; i8=int8([-1 2]); u16=uint16([1 65535]); i64=int64([-9e15 9e15]); u64=uint64([0 18446744073709551615]);
nd = reshape(1:24,[2 3 4]); sc = 3.5; ssing = single(2);
sa = struct('a',{1,'two',[3 4]}); st = struct('x',{{'p','q'}});
sdbl = struct('m', magic(3));
names = {'tess','big','sp','spc','cs','lg','lgs','cx','cxs','ec','ec0','emptycell','emptystruct','chm','i8','u16','i64','u64','nd','sc','ssing','sa','st','sdbl'};
save(fullfile(d,'t73.mat'), names{:}, '-v7.3');
save(fullfile(d,'t7.mat'), names{:}, '-v7');
% opaque
s = "hello"; dt = datetime(2020,1,1); tb = table([1;2],'VariableNames',{'v'});
save(fullfile(d,'opaque73.mat'),'sc','s','dt','tb','-v7.3');
% fields of unicode & big struct elems
save(fullfile(d,'utf73.mat'),'Comment','-v7.3'); save(fullfile(d,'utf7.mat'),'Comment','-v7');
end
