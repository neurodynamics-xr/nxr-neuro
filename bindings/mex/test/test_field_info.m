function test_field_info()
fprintf('[test_field_info] starting\n');
thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'nxr_compute.%s not found', mexext);
addpath(hits(1).folder); clear nxr_compute

% ── tangentVertex ─────────────────────────────────────────────────────────────
fi = nxr_compute('operators.fieldById', 'tangentVertex');
assert(strcmp(fi.bundle,         'tangent'),           'bundle');
assert(strcmp(fi.field_type,     'complex'),            'field_type');
assert(strcmp(fi.representation, 'intrinsic_complex'),  'representation');

% ── ambientVertexWorld ────────────────────────────────────────────────────────
av = nxr_compute('operators.fieldById', 'ambientVertexWorld');
assert(strcmp(av.bundle,         'ambient'),            'ambient bundle');
assert(strcmp(av.representation, 'world'),              'world repr');

% ── operatorInfo: input_field / output_field ──────────────────────────────────
oi = nxr_compute('operators.operatorById', 'laplaceBeltrami');
assert(strcmp(oi.input_field,  'scalarVertex'),         'op input_field');
assert(strcmp(oi.output_field, 'scalarVertex'),         'op output_field');

% ── unknown id must error ─────────────────────────────────────────────────────
try
    nxr_compute('operators.fieldById', 'nonExistentFieldXYZ');
    error('should have thrown');
catch ME
    assert(strcmp(ME.identifier, 'nxr:invalidInput'), ...
        'expected nxr:invalidInput, got %s', ME.identifier);
end

fprintf('test_field_info PASSED\n');
end
