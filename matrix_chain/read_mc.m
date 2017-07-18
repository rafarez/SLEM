% """
%
% Example:
%	out=read_mc('filename.mc')
%   [D,S,I]=out{:}; class(D), class(S), class(I)
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
function out=read_mc(fileName)

mc_htype='int';

%
fid=fopen(fileName, 'r');
if(fid<3), error('Unable to open file.'); end;

% Get end-of-file.
curr=ftell(fid); 
fseek(fid, 0, 'eof'); eof=ftell(fid); 
fseek(fid, curr, 'bof');

%
out={};
while(true)
    shape=fread(fid, 2, mc_htype);    
    shape=shape(:)'; shape=fliplr(shape); %The file is in row-major.
    shape = abs(shape);
    if feof(fid), break; end;    
    
    type=mapType(fread(fid, 1, 'char*1'));
    cM=fread_fail(fid, shape(:)', [type, '=>', type]);
    out{end+1}=cM'; %The file is in row-major.
end
fclose(fid);

%%
function out=mapType(npyType)
switch npyType
    case 'f', out='single';
    case 'd', out='double';
    case 'i', out='int';
    case 'I', out='uint';
    case 'b', out='int8';
    case 'B', out='uint8';
    otherwise, error(['Invalid input type: ', npyType])
end
%%
function out=fread_fail(fid, shape, varargin)
    [out, count]=fread(fid, shape, varargin{:});        
    if count~=prod(shape), error('Failed reading! The file is corrupt!'); end
  
%%