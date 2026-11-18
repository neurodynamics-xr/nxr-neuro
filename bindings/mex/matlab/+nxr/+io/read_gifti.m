function image = read_gifti(path)
%NXR.IO.READ_GIFTI  Read a GIFTI file (surfaces, shape and functional maps, label maps): a struct with version, metadata, labels and arrays.
%   image = nxr.io.read_gifti("lh.pial.surf.gii")
%   vertices = image.arrays(1).data;    % [n x 3] single
%   faces = image.arrays(2).data;       % [n x 3] int32, 1-based: straight into nxr.compute.operators.eigen
%
%   Each element of arrays is a struct: intent ("NIFTI_INTENT_POINTSET", "NIFTI_INTENT_TRIANGLE", "NIFTI_INTENT_SHAPE", ...), datatype, metadata,
%   transforms and data, an array of the file's dimensions whatever the file's indexing order (single, int32 or uint8). The vertex indices of a
%   triangle array are 1-based (the file's are 0-based; the Python wheel keeps them). ASCII, Base64Binary and GZipBase64Binary encodings are read in
%   either byte order; ExternalFileBinary is refused. Metadata keys are the file's, as valid MATLAB field names.
    arguments
        path {mustBeTextScalar}
    end
    image = nxr_compute('io.read_gifti', char(path));
end
