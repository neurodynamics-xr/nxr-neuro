function [data, header] = read_nifti(path)
%NXR.IO.READ_NIFTI  Read a NIfTI-1 or NIfTI-2 volume (.nii, .nii.gz, or a .hdr/.img pair named by either): [data, header].
%   [data, header] = nxr.io.read_nifti("T1.nii.gz")
%
%   data has the shape the file states, with the file's x axis first (MATLAB's own order, no transpose), scaled by scl_slope and scl_inter when the
%   file sets them. 8- and 16-bit integers and float32 read as single; 32- and 64-bit integers and float64 as double. Complex, RGB and 128-bit types are
%   refused. header holds every field of the file's header; header.affine is the 4 x 4 voxel to world map of the best form (sform, else qform, else the
%   pixel sizes; 0-based voxel indices as the file states them), in header.spatial_unit (unit_to_metres converts).
    arguments
        path {mustBeTextScalar}
    end
    [data, header] = nxr_compute('io.read_nifti', char(path));
end
