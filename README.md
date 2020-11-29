# Sudoku solver
Simple project solving 9x9 sudoku from file.
Project created for Communication Human-Computer classes.

### Installation (conda required):
- For new environment: `conda create --name <name> python=3.6 --file spec-file.txt ` 
- For existing environment: `conda install --name <name> --file spec-file.txt` 

### Running project 
- `conda activate <name>`
- Commands:
    - `python main.py solve <image_path_name>` - solve sudoku on image 
    - `python main.py test` - test complete sudoku solving program
    - `python main.py test_recognition` -  test number recognition from image
    - `python main.py test_solver <algorithm>` - test sudoku solving algorithm (`basic` or `possibilities`)
- `jupyter-notebook` for generating number recognition model

### Dependencies export:
- `conda list --explicit > spec-file-windows.txt`

Project created by:
- [@Cheriit](https://github.com/Cheriit/)
- [@roman-oberenkowski](https://github.com/roman-oberenkowski)
- [@ajana4096](https://github.com/ajana4096/)
