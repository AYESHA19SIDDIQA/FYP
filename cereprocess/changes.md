## Breaking Changes in v2

- **`v2`** is a major release that introduces breaking changes. The following changes are made in this release:

### Dataset.save_to_npz()

- The save_to_npz now uses a single argument(destdir) where it converts both train and eval without specification
- Instead of returning individual traindir and destdir, it returns destdir, from which we can extract traindir and destdir
- destdir/data.csv contains the combined data now


### Created _procces_files private function

- The function has been made to reuse the code in the .save_to_npz() function

