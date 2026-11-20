// The options of the commands that take an array (the same set as the Python face's ARRAY_OPTIONS, nxr_neuro/_commands.py): given as a typed array (or { data, shape, dtype? }),
// handed to the library in memory under the name the producer records (`values.npy`); a string is the path of a .npy file, as on the command line. `add measurements`'s
// `measure` is an object of arrays by measure name (or the command line's NAME=FILE.npy strings).
export const ARRAY_OPTIONS = {
  'add field': ['values'],
  'add selection': ['members', 'indices', 'spans'],
  'add statistic': ['values'],
  'add operator': ['values', 'data', 'indices', 'indptr'],
  'add points': ['positions'],
  'add surface': ['vertices', 'faces'],
};
