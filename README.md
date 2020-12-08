# Sudoku solver
Simple project solving 9x9 sudoku from file.
Project created for Communication Human-Computer classes.

### Installation (conda required):
- For new environment: `conda env create -f environment.yml ` 
- For existing environment: `conda env update --prefix ./env --file environment.yml --prune` 
- Env activation `conda activate SudokuSolver`
### Running project 
- `conda activate <name>`
- Commands:
    - `python main.py solve <image_path_name>` - solve sudoku on image 
    - `python main.py test` - test complete sudoku solving program
    - `python main.py test_recognition` -  test number recognition from image
    - `python main.py test_solver <algorithm>` - test sudoku solving algorithm (`basic` or `possibilities`)
    - `python main.py test_save_img` - save images for every stage of processing each input image (output generates to folder test_img) 
- `jupyter-notebook` for generating number recognition model

### Dependencies export:
- `conda env export > environment.yml`

Project created by:
- [@Cheriit](https://github.com/Cheriit/)
- [@roman-oberenkowski](https://github.com/roman-oberenkowski)
- [@ajana4096](https://github.com/ajana4096/)
