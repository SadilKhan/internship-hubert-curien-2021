# AutoLabelme

Autolabelme takes the json file created in LabelMe and then start template matching for every label.

## To Run
1. Open `Terminal`.
2. `labelme` or `python3 /path/to/labelme.py`. Check [here](https://github.com/wkentaro/labelme) for more details.
3. Create one bounding box per label.
4. Run AutoLabelme.py ` python3 /path/to/AutoLabelme.py`.
5. Open JSON and press `Next Line >>` for starting to match.
6. Press `Show All Boxes` for showing all the boxes that the algorithm has already generated.
7. Press `<< Previous Line` for viewing the matched boxes for the previous label.
8. Press `+` for more boxes and `-` for less boxes or box repositioning.
9. If you have rotated image, fill the rotation range or just enter `min` value. For example `min=45` and `max=90` will give `list(range(45,90,1))` values or just enter `min=45` which will only rotate the image once (45 degree).
10. Press `Add` button. Then press `Rematch` button.
11. Press  `Save JSON` to save a json file.
12. Open the save json file in Labelme. Labelme will show the matched templates. Edit it if necessary.
13. Press `Save Images` in AutoLabelme if all the boxes are okay. This will save the matched templates in JPEG.


## Function of every buttons:
1. `Next Line >>`: Template matching for next label
2. `<< Previous Line`: Template Matching for the previous label
3. `-`: Increases the threshold which results in less number of boxes.
4. `+`: Decreases the threshold which results in more number of boxes.
5. `Save JSON`: Saves a JSON file which can be read by LabelMe for further edits.
6. `Save Images`: Save the cropped vignettes from the image
7. `min`: The minimum value for Rotation.
8. `max`: The maximum value for Rotation.
9. `Rematch`: Match again for current label.

## Libraries Needed

1. Numpy (`pip install numpy`)
2. OpenCV(`pip install opencv-python`)
3. PIL (`pip install pillow`)
4. tkinter (`pip install tk`)
5. colorsys (`pip install colorsys`)
6. tkmacosx (`pip install tkmacosx`) --> (ONLY FOR MAC USERS)
7. labelme (`pip install labelme`)[WINDOWS/LINUX] (`brew install labelme`)[MAC]

## The Other Py files
1. `templatematcher.py` --> Matches templates for specified labels or all the labels. Requires JSON file created using Labelme.
2. `templatesaver.py`--> Saves the matched images.
