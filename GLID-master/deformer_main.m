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
imgfilepath  = fullfile(datadir, 'ball.png');

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
    offset = 10;
    simplify = 20;
    allcages = GetCompCage(imgfilepath, offset, simplify, 0, 1);%lsb 0->1
end

cage = allcages{1};
holes = reshape( allcages(2:end), 1, [] );

[im,~,mask]=imread(imgfilepath);
figure; image(im); hold on; axis equal;
[w, h, ~] = size(im);
cellfun(@(x) plot( conj(x([1:end 1])*100+h/2-w/2*1i), 'r', 'linewidth', 2 ), allcages)

[X, T] = cdt([cage, holes], [], numMeshVertex, false);
X = fR2C(X);

%% load p2p
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
