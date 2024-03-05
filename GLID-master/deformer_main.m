fC2R = @(x) [real(x) imag(x)];
fR2C = @(x) complex(x(:,1), x(:,2));

%%
if ~exist('working_dataset', 'var') || isempty(working_dataset)
    working_dataset = 'rex';
    fprintf('working shape default: %s\n', working_dataset);
end

datadir = fullfile(cd, 'data', working_dataset, '\');

if exist('numMeshVertex', 'var')~=1 
    numMeshVertex = 10000;
    fprintf('numMeshVertex default: %d\n', numMeshVertex);
end



datafile = fullfile(datadir, 'data.mat');
imgfilepath  = fullfile(datadir, 'ball1.png');
%imgfilepath  = fullfile(datadir, 'image.png');

P2Psrc = zeros(0,1); P2Pdst = zeros(0,1);

if exist(datafile, 'file') == 3%lsb2->3
    %% load presaved data
    load(datafile);
else    
    % read image dimension
    iminfo = imfinfo(imgfilepath);
    img_w = iminfo.Width;
    img_h = iminfo.Height;
    
    %% extract cage from image, only simply-connected domain is supported
    offset = 100;
    simplify = 100;
    allcages = GetCompCage(imgfilepath, offset, simplify, 0, 1);%lsb 0->1
end

cage = allcages{1};
holes = reshape( allcages(2:end), 1, [] );

% [im,~,mask]=imread(imgfilepath);
% figure; image(im); hold on; axis equal;
% [w, h, ~] = size(im);
% cellfun(@(x) plot( conj(x([1:end 1])*100+h/2-w/2*1i), 'r', 'linewidth', 2 ), allcages)

[X, T] = cdt([cage, holes], [], numMeshVertex, false);
X = fR2C(X);

X1=fC2R(X);
trisurf(T, X1(:,1), X1(:,2), zeros(size(X1,1),1)) % Assuming X is a 2-column matrix of [x, y] coordinates
axis equal
xlabel('X Coordinate')
ylabel('Y Coordinate')
zlabel('Z Coordinate') % Even though it's a 2D triangulation, trisurf expects a Z coordinate which we set to zero.
title('Triangulation of Given Points')
%% load p2p
% P2Psrc=[-14.324579635372867 + 1.713159374569387i ;
% -6.028442372470105 + 0.8239714274935914i ;
% 0.11523202970772446 + -1.1908044809238392i; 
% -2.8116128002061256 + -4.302495005528787i ;
% 0.40176693990017864 + -4.383133658931498i ;
% 4.952812142767244 + 2.8249548543328187i ;
% -1.9490123939987198 + -7.298937729995031i; 
% -3.772056682884389 + -7.3045776811116845i ];
% 
% P2Pdst=[-8.131714820861816 + -4.065834999084473i ;
% -7.596327781677246 + 1.6159509420394897i ;
% -1.9128398895263672 + -0.23459529876708984i; 
% -7.25319242477417 + -3.4069440364837646i ;
% -2.601797580718994 + -3.3349080085754395i ;
% 2.8360509872436523 + 0.18369722366333008i ;
% -5.329527378082275 + -6.550514221191406i ;
% -6.821569442749023 + -6.493046760559082i 
% ];

P2PVtxIds = triangulation(T, fC2R(X)).nearestNeighbor( fC2R(P2Psrc) );
P2PCurrentPositions = P2Pdst;
iP2P = 1;

%% texture
uv = fR2C([real(X)/img_w imag(X)/img_h])*100 + complex(0.5, 0.5);

%% solvers & energies
harmonic_map_solvers = {'AQP', 'SLIM', 'Newton', 'Newton_SPDH', 'Newton_SPDH_FullEig', 'Gradient Descent', 'LBFGS', ...
                        'cuGD', 'cuNewton', 'cuNewton_SPDH', 'cuNewton_SPDH_FullEig'};

if hasGPUComputing                    
    default_harmonic_map_solver = 'cuNewton_SPDH';
else
    warning('no cuda capable GPU present, switching to CPU solver');
    default_harmonic_map_solver = 'Newton_SPDH';
end

harmonic_map_energies = {'SymmDirichlet', 'Exp_SymmDirichlet', 'AMIPS'};
default_harmonic_map_energy = 'SymmDirichlet';
