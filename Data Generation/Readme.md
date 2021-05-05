# AutoLabelme

Autolabelme takes the json file created in LabelMe and then start template matching for every label.

## To Run
1. Open `Terminal`.
2. `labelme` or `python3 /path/to/labelme.py`. Check [here](https://github.com/wkentaro/labelme) for more details.
3. Create one bounding box per label.
4. Run AutoLabelme.py ` python3 /path/to/AutoLabelme.py`
5. Open JSON and start `Start Matching`.
6. Press `Next >>` for starting matching for the next label
7. Press `<< Previous` for viewing the matched boxes for the previous label
8. Press `Less Boxes` if false positives are detected otherwise `Resize Boxes` and `Finer Resize Boxes` for perfectly repositioning anf resizing the boxes.
9. Press `More Boxes` if true positive is very low.
10. Press  `Save JSON` to save a json file.
11. Open the save json file in Labelme. Labelme will show the matched templates. Edit it if necessary.
12. Press `Save Images` in AutoLabelme if all the boxes are okay. This will save the matched templates in JPEG.
## To save the cropped images/vignettes
`python3 /path/to/templatesaver.py <json file>`

## Function of every buttons:
1. Next: Template matching for next label
2. Previous: Template Matching for the previous label
3. Less Boxes: Increases the threshold which results in less number of boxes.
4. More Boxes: Decreases the threshold which results in more number of boxes.
5. Resize Boxes: Increases the threshold which results in resizing the boxes or reduce the number.
6. Finer Resize Boxes: Increases the threshold by little which results in finer resizing the boxes.
7. Save JSON: Saves a JSON file which can be read by LabelMe for further edits.
8. Save Images: Save the cropped vignettes from the image

## Libraries Needed

1. Numpy (`pip install numpy`)
2. PIL (`pip install pillow`)
3. tkinter (`pip install tkinter`)
4. colorsys (`pip install colorsys`)
5. tkmacosx (`pip install tkmacosx`) --> (ONLY FOR MAC USERS)
6. labelme (`pip install labelme`)[WINDOWS/LINUX] (`brew install labelme`)[MAC]

## The Other Py files
1. `templatematcher.py` --> Matches templates for specified labels or all the labels. Requires JSON file created using Labelme.
2. `templatesaver.py`--> Saves the matched images.
