% """
%Example:
%	D=double(reshape(1:10, [5,2]))
%	S=single(reshape(1:10, [5,2]))
%	I=int32(reshape(1:10, [5,2]))
%	write_mc('filename.mc', D,S,I)
%
% Read/write multiple matrices to a file in binary format. The file-stream format is:
% M1,M2,M3,...

% Where each M stream consists of the following data:
% R,C,[data type specifier character]
% [d11]...[d1C]
% ...
% [dR1]...[dRC]

% Valid data type specifiers (numpy character specifier convention):
%   f,d: 32-bit and 64-bit floating-point.
%   i,I: 32-bit signed and unsigned integers.
%   b,B: 8-bit signed and unsigned integers.
% """

%%
function write_mc(fileName, varargin)

mc_htype='int';

%Check data
for k=1:numel(varargin)
    M=varargin{k};
    if ~ismember(class(M), ['single', 'double', 'int', 'uint', 'int8', 'uint8'])
        error(['Invalid data type', class(M)]);
    end
end

%Write data
fid=fopen(fileName, 'w');
if(fid<3), error('Unable to open file.'); end;
for k=1:numel(varargin)
    M=varargin{k};
    fwrite_fail(fid, size(M), mc_htype);
    fwrite_fail(fid, mapType(class(M)), 'char*1');
    fwrite_fail(fid, M', class(M)) %The file is in row-major.
end;

%Close file.
fclose(fid);

%%
function out=mapType(matType)
switch matType
    case 'single', out='f';
    case 'double', out='d';
    case 'int32', out='i';
    case 'uint32', out='I';
    case 'int8', out='b';
    case 'uint8', out='B';
    otherwise, error(['Invalid input type: ', matType])
end 

%%
function fwrite_fail(fid, M, varargin)
    count=fwrite(fid, M, varargin{:});
    if count~=numel(M), error('Failed writing! The file is corrupt!'); end
    
%%